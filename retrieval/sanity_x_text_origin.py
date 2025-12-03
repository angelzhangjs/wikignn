#!/usr/bin/env python3
"""
Sanity check: do node features x come from real text or from ID-like strings?

For a given node type (default: 'entity'), this script:
  - Loads the graph (.pt) and associated metadata maps
  - Gathers per-node text (entity_text_en_by_index -> labels -> node_index inversion)
  - Encodes text and, if available, the node's ID string using the chosen backend (SBERT/CLIP)
  - Computes cosine(x, text) vs cosine(x, id_string)
  - Reports summary statistics (mean/median), and fraction of ID-like texts

Example:
  python3 retrieval/sanity_x_text_origin.py \
    --path graph_embedding/graph_by_label.no_literal.pyg.pt.with_clip.pt \
    --node-type entity \
    --backend sbert --sbert-model sentence-transformers/all-MiniLM-L6-v2 \
    --num-samples 200
"""
from __future__ import annotations

import argparse
import re
import random
from typing import Dict, List, Tuple

import torch

try:
    from torch_geometric.data import HeteroData  # type: ignore
except Exception:
    HeteroData = None  # type: ignore

# Backends
try:
    import clip  # type: ignore
except Exception:
    clip = None
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to .pt (HeteroData or dict with 'pyg_data')")
    ap.add_argument("--node-type", default="entity", help="Node type to analyze (default: entity)")
    ap.add_argument("--backend", choices=["sbert", "clip"], default="sbert", help="Text encoder backend")
    ap.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2", help="SBERT model name")
    ap.add_argument("--clip-model", default="ViT-B/32", help="CLIP model name (e.g., ViT-B/32, RN50)")
    ap.add_argument("--num-samples", type=int, default=100, help="Number of nodes to sample")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--max-chars", type=int, default=512, help="Cap text length")
    return ap.parse_args()


def load_graph(path: str) -> Tuple[torch.nn.Module, Dict]:
    obj = torch.load(path, map_location="cpu")
    meta: Dict = obj if isinstance(obj, dict) else {}
    data = meta.get("pyg_data", obj)
    return data, meta


def get_texts_and_ids(meta: Dict, ntype: str, num_nodes: int, max_chars: int) -> Tuple[List[str], List[str]]:
    # texts: prefer rich text, then labels, else empty
    texts = [""] * num_nodes
    ids = [""] * num_nodes
    if isinstance(meta, dict):
        # text by index
        if f"{ntype}_text_en_by_index" in meta and isinstance(meta[f"{ntype}_text_en_by_index"], dict):
            m = meta[f"{ntype}_text_en_by_index"]
            for k, v in m.items():
                try:
                    i = int(k)
                    s = str(v).replace("\n", " ").strip()
                    if max_chars and len(s) > max_chars:
                        s = s[:max_chars]
                    if 0 <= i < num_nodes:
                        texts[i] = s
                except Exception:
                    continue
        # labels fallback
        elif f"{ntype}_labels_en" in meta and isinstance(meta[f"{ntype}_labels_en"], dict):
            m = meta[f"{ntype}_labels_en"]
            for k, v in m.items():
                try:
                    i = int(k)
                    s = str(v).replace("\n", " ").strip()
                    if max_chars and len(s) > max_chars:
                        s = s[:max_chars]
                    if 0 <= i < num_nodes:
                        texts[i] = s
                except Exception:
                    continue
        # node_index inversion for IDs
        if "node_index" in meta and isinstance(meta["node_index"], dict) and ntype in meta["node_index"]:
            inv = meta["node_index"][ntype]
            if isinstance(inv, dict):
                for id_str, idx in inv.items():
                    try:
                        i = int(idx)
                        if 0 <= i < num_nodes:
                            ids[i] = str(id_str)
                    except Exception:
                        continue
    return texts, ids


def embed_texts_sbert(texts: List[str], model_name: str) -> torch.Tensor:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")
    model = SentenceTransformer(model_name, device="cpu")
    embs = model.encode([t if t is not None else "" for t in texts],
                        convert_to_tensor=True, normalize_embeddings=True).float()
    return embs.cpu()  # [N, D]


def embed_texts_clip(texts: List[str], model_name: str) -> torch.Tensor:
    if clip is None:
        raise RuntimeError("CLIP not installed. pip install git+https://github.com/openai/CLIP.git")
    device = "cpu"
    model, _ = clip.load(model_name, device=device)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 512):
            batch = [ (t or "").replace("\n", " ").strip() for t in texts[i:i+512] ]
            tokens = clip.tokenize(batch, truncate=True).to(device)
            feats = model.encode_text(tokens).float()
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
            out.append(feats.cpu())
    return torch.cat(out, dim=0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-12)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-12)
    return (a * b).sum(dim=-1)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data, meta = load_graph(args.path)
    if HeteroData is not None and isinstance(data, HeteroData):
        if args.node_type not in data.node_types:
            raise SystemExit(f"Node type '{args.node_type}' not found. Available: {data.node_types}")
        x = data[args.node_type].x
        num_nodes = int(data[args.node_type].num_nodes)
    else:
        # Homogeneous fallback
        x = getattr(data, "x", None)
        if x is None:
            raise SystemExit("No node features found on homogeneous graph.")
        num_nodes = x.size(0)

    texts, ids = get_texts_and_ids(meta, args.node_type, num_nodes, args.max_chars)
    idxs = [i for i in range(num_nodes) if i < x.size(0)]
    if not idxs:
        raise SystemExit("No nodes to sample.")
    random.shuffle(idxs)
    idxs = idxs[: min(args.num_samples, len(idxs))]

    # Build text embeddings for sampled nodes
    sampled_texts = [texts[i] for i in idxs]
    sampled_ids = [ids[i] for i in idxs]

    if args.backend == "sbert":
        t_emb = embed_texts_sbert(sampled_texts, args.sbert_model)
        id_emb = embed_texts_sbert(sampled_ids, args.sbert_model)
    else:
        t_emb = embed_texts_clip(sampled_texts, args.clip_model)
        id_emb = embed_texts_clip(sampled_ids, args.clip_model)

    x_emb = x[idxs].float().cpu()
    # If x dim != text dim, project to common min-dim for a rough check (or skip)
    if x_emb.size(1) != t_emb.size(1):
        d = min(x_emb.size(1), t_emb.size(1))
        x_emb = x_emb[:, :d]
        t_emb = t_emb[:, :d]
        id_emb = id_emb[:, :d]

    cos_text = cosine(x_emb, t_emb).tolist()
    cos_id = cosine(x_emb, id_emb).tolist()

    # ID-like heuristic
    id_like = [bool(re.fullmatch(r"[A-Z]\d+", s or "")) for s in sampled_texts]
    frac_id_like = sum(id_like) / max(1, len(id_like))

    import numpy as np
    def stats(a: List[float]):
        return float(np.mean(a)), float(np.median(a))

    m_text, med_text = stats(cos_text)
    m_id, med_id = stats(cos_id)

    print(f"Samples: {len(idxs)}  node_type: {args.node_type}  backend: {args.backend}")
    print(f"cos(x, text): mean={m_text:.3f} median={med_text:.3f}")
    print(f"cos(x,  id ): mean={m_id:.3f} median={med_id:.3f}")
    print(f"ID-like text fraction (regex [A-Z]\\d+): {frac_id_like:.3f}")
    if med_text > med_id + 0.05 and frac_id_like < 0.3:
        print("Inference: x likely comes from real text embeddings.")
    elif abs(med_text - med_id) < 0.02 and frac_id_like > 0.5:
        print("Inference: x may be derived from ID-like strings (or labels fell back to IDs).")
    else:
        print("Inference: mixed signal; check your text maps and embedding backend.")


if __name__ == "__main__":
    main()


