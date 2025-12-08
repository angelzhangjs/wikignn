#!/usr/bin/env python3
from __future__ import annotations
"""
Load a trained dual-encoder checkpoint, retrieve top-K candidate subgraphs for a query,
and optionally ask Gemini to answer using the retrieved context.
"""
import argparse
import glob
import os
from typing import List, Tuple, Dict, Set

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch  # type: ignore

from retrieval.train_gnn_pairs import TextEncoder, HeteroSubgraphEncoder  # reuse model defs
from retrieval.pairs_dataset import _load_subgraph
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

def _load_ckpt(ckpt_path: str, device: torch.device) -> Tuple[dict, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or "model_gnn" not in ckpt or "model_txt" not in ckpt:
        raise RuntimeError(f"Unexpected checkpoint format at {ckpt_path}")
    cfg = ckpt.get("config", {})
    return ckpt, cfg

def _build_models_from_config(cfg: dict, device: torch.device) -> Tuple[TextEncoder, HeteroSubgraphEncoder]:
    text_backend = cfg.get("text_backend", "sbert")
    sbert_model = cfg.get("sbert_model", "sentence-transformers/all-MiniLM-L6-v2")
    clip_model = cfg.get("clip_model", "ViT-B/32")
    out_dim = int(cfg.get("out_dim", 256))
    finetune_text = bool(cfg.get("finetune_text", False))
    use_edge_prompt = bool(cfg.get("use_edge_prompt", False))
    edge_prompt_dim = int(cfg.get("edge_prompt_dim", 256))
    edge_prompt_anchors = int(cfg.get("edge_prompt_anchors", 32))

    model_txt = TextEncoder(
        backend=text_backend,
        sbert_model=sbert_model,
        clip_model=clip_model,
        out_dim=out_dim,
        finetune_text=finetune_text,
    ).to(device)

    # During inference we won't know node dims/edge types until the first batch;
    # HeteroSubgraphEncoder lazily builds layers on first forward.
    model_gnn = HeteroSubgraphEncoder(
        out_dim=out_dim,
        dropout=0.1,
        use_edge_prompt=use_edge_prompt,
        edge_prompt_dim=edge_prompt_dim,
        edge_prompt_anchors=edge_prompt_anchors,
        predefined_edge_types=None,
        predefined_node_dims=None,
    ).to(device)

    return model_txt, model_gnn

def _scan_candidates_union_metadata(candidate_paths: List[str]) -> Tuple[Tuple[Tuple[str, str, str], ...], Dict[str, int]]:
    """
    Scan candidates to collect the union of edge types and node feature dims.
    Returns:
      - union_edge_types: tuple of (src, rel, dst)
      - node_dims: dict mapping node_type -> feature_dim
    """
    edge_types_set: Set[Tuple[str, str, str]] = set()
    node_dims: Dict[str, int] = {}
    for p in candidate_paths:
        try:
            obj = torch.load(p, map_location="cpu")
            subg = obj["pyg_data"] if isinstance(obj, dict) and "pyg_data" in obj else obj
        except Exception:
            continue
        try:
            for et in subg.edge_types:
                edge_types_set.add(et)
            for nt in subg.node_types:
                x = getattr(subg[nt], "x", None)
                if x is not None:
                    d = int(x.size(-1))
                    if d > 0:
                        if nt in node_dims and node_dims[nt] != d:
                            # Prefer the first seen; dims should be consistent across graph
                            pass
                        else:
                            node_dims[nt] = d
        except Exception:
            continue
    union_edge_types = tuple(sorted(list(edge_types_set)))
    return union_edge_types, node_dims 

@torch.no_grad()
def embed_candidates(
    model_gnn: HeteroSubgraphEncoder,
    candidate_paths: List[str],
    device: torch.device,
    batch_size: int = 1,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Returns:
      - candidate_embeddings: [N, D] tensor on CPU
      - ordered_paths: list[str] aligned with candidate_embeddings
    """
    embs: List[torch.Tensor] = []
    ordered_paths: List[str] = []
    iterator = candidate_paths
    if tqdm is not None:
        iterator = tqdm(candidate_paths, desc="Embedding candidates", unit="cand")
    for p in iterator:
        try:
            obj = torch.load(p, map_location="cpu")
            subg = obj["pyg_data"] if isinstance(obj, dict) and "pyg_data" in obj else obj
            seed_node_type = obj.get("seed_node_type", None) if isinstance(obj, dict) else None
            seed_local_index = obj.get("seed_local_index", -1) if isinstance(obj, dict) else -1
            if seed_node_type is None:
                # Fallback: default to 'entity' and try to locate via n_id==global_id if available
                seed_node_type = "entity"
            if seed_local_index is None or seed_local_index < 0:
                try:
                    seed_global_id = int(obj.get("seed_global_id", -1)) if isinstance(obj, dict) else -1
                    n_id = subg[seed_node_type].n_id
                    where = (n_id == seed_global_id).nonzero(as_tuple=False)
                    if where.numel() > 0:
                        seed_local_index = int(where.view(-1)[0].item())
                except Exception:
                    seed_local_index = -1
        except Exception:
            continue
        hb = Batch.from_data_list([subg]).to(device)
        if seed_local_index >= 0:
            g = model_gnn(hb, seed_info=(seed_node_type, seed_local_index))  # [D]
        else:
            g = model_gnn(hb)  # fallback to pooled embedding
        embs.append(g.detach().cpu().unsqueeze(0))
        ordered_paths.append(p)
    if not embs:
        return torch.empty(0, 0), []
    return torch.cat(embs, dim=0), ordered_paths


@torch.no_grad()
def embed_query(model_txt: TextEncoder, query: str, device: torch.device) -> torch.Tensor:
    q = model_txt([query])  # [1, D]
    return F.normalize(q, dim=-1).cpu()

def summarize_subgraph(path: str) -> str:
    """
    Produce a lightweight textual summary of a candidate subgraph for LLM context.
    """
    try:
        subg = _load_subgraph(path)
    except Exception:
        return f"Subgraph: {os.path.basename(path)} (failed to load)"
    parts: List[str] = [f"Subgraph: {os.path.basename(path)}"]
    parts.append("Node types:")
    for nt in subg.node_types:
        nn = int(subg[nt].num_nodes)
        xd = int(subg[nt].x.size(-1)) if hasattr(subg[nt], "x") and subg[nt].x is not None else 0
        parts.append(f"  - {nt}: {nn} nodes (x_dim={xd})")
    parts.append("Edges:")
    for et in subg.edge_types:
        ei = subg[et].edge_index
        ecount = 0 if ei is None else int(ei.size(1))
        ed = 0
        if hasattr(subg[et], "edge_attr") and subg[et].edge_attr is not None:
            ed = int(subg[et].edge_attr.size(-1))
        parts.append(f"  - {et}: {ecount} edges (edge_attr_dim={ed})")
    return "\n".join(parts)


def ask_gemini(query: str, contexts: List[str], api_key: str | None, model_name: str, max_contexts: int | None = None) -> str:
    if not api_key:
        return "Gemini API key not provided. Skipping LLM answer."
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        return "google-generativeai not installed. pip install google-generativeai"
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = (
        "You are given a user query and several graph subgraph summaries.\n"
        "Use the summaries as context to answer. If unsure, say so briefly.\n\n"
        f"Query:\n{query}\n\n"
        "Context:\n" + "\n\n".join(contexts[:max_contexts] if max_contexts is not None else contexts)
    )
    try:
        resp = model.generate_content(prompt)
    except Exception as e:
        return f"Gemini error: {e}"
    try:
        return resp.text or "(no text in response)"
    except Exception:
        return str(resp)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Retrieve top-K subgraphs and optionally ask Gemini")
    ap.add_argument("--checkpoint", required=True, help="Path to best.pt/.pth checkpoint")
    ap.add_argument("--candidates-dir", required=True, help="Directory of candidate subgraph .pt files")
    ap.add_argument("--query", required=True, help="User query text")
    ap.add_argument("--topk", type=int, default=5, help="Top-K candidates to return")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gemini", action="store_true", help="Call Gemini with retrieved context")
    ap.add_argument("--gemini-model", default="gemini-1.5-flash", help="Gemini model name")
    ap.add_argument("--gemini-api-key", default=os.environ.get("GEMINI_API_KEY", ""), help="Gemini API key")
    ap.add_argument("--answer-out", default="", help="If set, save Gemini final answer to this .txt path")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    ckpt, cfg = _load_ckpt(args.checkpoint, device)
    model_txt, _ = _build_models_from_config(cfg, device)
    # Load text encoder weights directly (no lazy layers)
    model_txt.load_state_dict(ckpt["model_txt"])
    model_txt.eval()

    # Discover candidates before loading GNN weights, so we can build layers lazily
    cand_paths = sorted(glob.glob(os.path.join(args.candidates_dir, "*.pt")))
    if not cand_paths:
        print(f"No candidates found in {args.candidates_dir}")
        return 2

    # Scan candidates to collect union metadata and instantiate GNN with predefined types/dims
    union_edge_types, node_dims = _scan_candidates_union_metadata(cand_paths)
    model_gnn = HeteroSubgraphEncoder(
        out_dim=int(cfg.get("out_dim", 256)),
        dropout=0.1,
        use_edge_prompt=bool(cfg.get("use_edge_prompt", False)),
        edge_prompt_dim=int(cfg.get("edge_prompt_dim", 256)),
        edge_prompt_anchors=int(cfg.get("edge_prompt_anchors", 32)),
        predefined_edge_types=union_edge_types,
        predefined_node_dims=node_dims if node_dims else None,
    ).to(device)
    # Build layers with one dummy forward on the first candidate; convs are created for union via predefined types
    first_obj = torch.load(cand_paths[0], map_location="cpu")
    first_subg = first_obj["pyg_data"] if isinstance(first_obj, dict) and "pyg_data" in first_obj else first_obj
    hb0 = Batch.from_data_list([first_subg]).to(device)
    _ = model_gnn(hb0)
    # Try strict load first; if it fails due to relation/dim mismatch, fall back to relaxed load
    try:
        model_gnn.load_state_dict(ckpt["model_gnn"], strict=True)
    except Exception as e:
        print(f"Strict load failed ({e}); falling back to relaxed weight loading.")
        missing, unexpected = model_gnn.load_state_dict(ckpt["model_gnn"], strict=False)
        print(f"Relaxed load completed (missing={len(missing)}, unexpected={len(unexpected)})")
    model_gnn.eval()

    cand_embs, ordered = embed_candidates(model_gnn, cand_paths, device=device, batch_size=1)
    if cand_embs.numel() == 0:
        print("Failed to embed candidates (no valid subgraphs).")
        return 3
    q = embed_query(model_txt, args.query, device=device)  # [1, D]
    scores = (cand_embs @ q.squeeze(0))  # [N]
    topk = min(args.topk, scores.size(0))
    top_scores, top_idx = torch.topk(scores, k=topk, largest=True, sorted=True)
    top_paths = [ordered[int(i)] for i in top_idx]

    print("Top-K retrieved subgraphs:")
    for rank, (p, s) in enumerate(zip(top_paths, top_scores.tolist()), start=1):
        print(f"{rank:2d}. {os.path.basename(p)}  score={s:.4f}")

    if args.gemini:
        contexts = [summarize_subgraph(p) for p in top_paths]
        answer = ask_gemini(
            args.query,
            contexts,
            api_key=args.gemini_api_key,
            model_name=args.gemini_model,
            max_contexts=args.topk,
        )
        print("\nGemini answer:\n")
        print(answer)
        if args.answer_out:
            try:
                out_dir = os.path.dirname(args.answer_out)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                with open(args.answer_out, "w", encoding="utf-8") as f:
                    f.write(answer if isinstance(answer, str) else str(answer))
                print(f"\nSaved Gemini answer to {args.answer_out}")
            except Exception as e:
                print(f"Failed to save Gemini answer to {args.answer_out}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


