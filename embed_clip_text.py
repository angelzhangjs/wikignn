#!/usr/bin/env python3
"""
Create CLIP text embeddings for:
- Nodes/entities: entity_text_{lang}_by_index -> clip_text_node_emb_{lang} (Tensor [N, D])
- Relation labels (per PID): property_labels_{lang} -> clip_text_rel_emb_{lang}_by_pid (dict pid -> D-vector)
- Edges (mean per labeled relation): edge_text_pairs_label_{lang} -> clip_text_edge_mean_emb_{lang} (dict rel_label_key -> D-vector)

Usage:
  python embed_clip_text.py \
    --pt_in /home/ghr/angel/gnn/graph_output/clean_graph.pyg_en_labeled.pt \
    --lang en
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Tuple, Any

import torch

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Embed entity/relation/edge texts into CLIP vectors and save into .pt payload.")
    ap.add_argument("--pt_in", required=True, help="Path to input labeled graph .pt (e.g., clean_graph.pyg_en_labeled.pt)")
    ap.add_argument("--pt_out", default="", help="Path to save output .pt (default: overwrite input)")
    ap.add_argument("--lang", default="en", help="Language code (default: en)")
    ap.add_argument("--batch", type=int, default=256, help="Batch size for CLIP encoding")
    ap.add_argument("--edge_max_per_rel", type=int, default=2000, help="Cap number of edge text pairs per labeled relation")
    ap.add_argument("--model", default="ViT-B-32", help="open_clip model name")
    ap.add_argument("--pretrained", default="openai", help="open_clip pretrained tag")
    return ap.parse_args()


def _require_open_clip():
    try:
        import open_clip  # type: ignore
        return open_clip
    except Exception as e:
        raise RuntimeError(
            "open-clip-torch is required. Install with: python -m pip install open-clip-torch"
        ) from e


@torch.no_grad()
def encode_texts(model, tokenizer, device: str, texts: List[str], batch: int) -> torch.Tensor:
    if not texts:
        # Determine output dim from model
        dim = model.text_projection.shape[-1]
        return torch.zeros(0, dim, dtype=torch.float32)
    outs: List[torch.Tensor] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        tok = tokenizer(chunk).to(device)
        feats = model.encode_text(tok)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        outs.append(feats.float().cpu())
    return torch.cat(outs, dim=0)


def main() -> int:
    args = parse_args()
    p = torch.load(args.pt_in, map_location="cpu")
    if not isinstance(p, dict):
        raise TypeError("Expected dict payload (.pt).")

    open_clip = _require_open_clip()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(args.model)

    # 1) Node/entity text embeddings
    text_by_idx_key = f"entity_text_{args.lang}_by_index"
    ent_text = p.get(text_by_idx_key, {})
    if not isinstance(ent_text, dict) or not ent_text:
        raise RuntimeError(f"Missing {text_by_idx_key} in {args.pt_in}. Generate entity metadata first.")
    try:
        max_idx = max(int(i) for i in ent_text.keys())
    except Exception:
        max_idx = max(int(k) for k in range(len(ent_text)))  # fallback
    node_texts = [ent_text.get(i, "") for i in range(max_idx + 1)]
    node_emb = encode_texts(model, tokenizer, device, node_texts, args.batch)  # [N, D]
    p[f"clip_text_node_emb_{args.lang}"] = node_emb.half()

    # 2) Relation label embeddings (per PID)
    pid_to_lbl = p.get(f"property_labels_{args.lang}", {})
    rel_lbl_texts: List[str] = []
    rel_lbl_index: List[str] = []
    if isinstance(pid_to_lbl, dict):
        for pid, lbl in pid_to_lbl.items():
            rel_lbl_index.append(str(pid))
            rel_lbl_texts.append(str(lbl))
    rel_lbl_emb = encode_texts(model, tokenizer, device, rel_lbl_texts, args.batch)
    p[f"clip_text_rel_emb_{args.lang}_by_pid"] = {pid: rel_lbl_emb[i].half() for i, pid in enumerate(rel_lbl_index)}

    # 3) Edge text embeddings (mean per labeled relation)
    et_pairs_key = f"edge_text_pairs_label_{args.lang}"
    et_pairs = p.get(et_pairs_key, {})
    edge_mean: Dict[str, torch.Tensor] = {}
    if isinstance(et_pairs, dict) and et_pairs:
        for rel_key, pairs in et_pairs.items():
            if not isinstance(pairs, list) or len(pairs) == 0:
                continue
            # Cap per relation to control cost
            pairs = pairs[: args.edge_max_per_rel] if args.edge_max_per_rel > 0 else pairs
            # Build "src [REL] label [REL] dst" texts for robustness
            # When rel_key is "srcType:Label:dstType"
            try:
                src_t, label, dst_t = str(rel_key).split(":")
                rel_mid = label
            except Exception:
                rel_mid = str(rel_key)
            texts = [f"{str(s)} [REL] {rel_mid} [REL] {str(d)}" for (s, d) in pairs]
            emb = encode_texts(model, tokenizer, device, texts, args.batch)  # [E, D]
            if emb.shape[0] > 0:
                edge_mean[rel_key] = emb.mean(dim=0).half()
    p[f"clip_text_edge_mean_emb_{args.lang}"] = edge_mean

    # Meta
    p.setdefault("clip_meta", {})["text_model"] = f"{args.model}-{args.pretrained}"

    out = args.pt_out or args.pt_in
    torch.save(p, out)
    print(f"Saved CLIP embeddings to {out}")
    print(f"Nodes: {tuple(node_emb.shape)} | Rel labels: {len(rel_lbl_index)} | Edge label means: {len(edge_mean)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


