#!/usr/bin/env python3
"""
Build a PyG HeteroData graph whose edge types use property textual labels
instead of PIDs, using the label-keyed edge indices produced earlier.

Inputs:
- A .pt payload that contains 'edge_index_label_{lang}' created by
  fetch_property_labels.py (e.g., edge_index_label_en).
- Optionally, top-level node embeddings like clip_text_node_emb_en to attach
  to the 'entity' node type when sizes match.

Outputs:
- Saves a HeteroData object (.pt) with relation names as labels.

Example:
  python build_graph_by_label.py \
    --in graph_output/clean_graph.pyg_en_labeled.en_labeled.pt \
    --lang en \
    --out graph_output/graph_by_label.pyg.pt
"""
from __future__ import annotations

import argparse
from typing import Dict, Tuple

import torch
from torch_geometric.data import HeteroData


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build HeteroData using label-valued edges")
    ap.add_argument("--in", dest="inp", required=True, help="Input .pt with edge_index_label_{lang}")
    ap.add_argument("--lang", default="en", help="Language code used for labels (default: en)")
    ap.add_argument("--out", required=True, help="Output .pt path for HeteroData")
    return ap.parse_args()


def ensure_tensor_2xE(value) -> torch.Tensor:
    t = torch.as_tensor(value)
    if t.dim() == 2 and t.size(0) == 2:
        return t.to(torch.long).contiguous()
    if t.dim() == 2 and t.size(1) == 2:
        return t.t().to(torch.long).contiguous()
    raise ValueError(f"Edge index has unexpected shape {tuple(t.shape)}; expected [2,E] or [E,2].")


def build_hetero_from_label_edges(payload: Dict, lang: str) -> HeteroData:
    key = f"edge_index_label_{lang}"
    if key not in payload or not isinstance(payload[key], dict):
        raise KeyError(f"Missing '{key}' in payload; run fetch_property_labels.py first.")
    edge_map: Dict[str, torch.Tensor] = payload[key]  # "src:label:dst" -> Tensor[2, E]

    data = HeteroData()
    # Track num_nodes per node type
    max_id_by_ntype: Dict[str, int] = {}

    for composite, ei in edge_map.items():
        parts = str(composite).split(":")
        if len(parts) != 3:
            # Skip malformed keys
            continue
        src_t, label, dst_t = parts
        edge_index = ensure_tensor_2xE(ei)
        data[(src_t, label, dst_t)].edge_index = edge_index
        # Track maxima for num_nodes inference
        if edge_index.numel() > 0:
            s_max = int(edge_index[0].max().item())
            d_max = int(edge_index[1].max().item())
            max_id_by_ntype[src_t] = max(s_max, max_id_by_ntype.get(src_t, -1))
            max_id_by_ntype[dst_t] = max(d_max, max_id_by_ntype.get(dst_t, -1))

    # Set num_nodes per node type
    for ntype, m in max_id_by_ntype.items():
        data[ntype].num_nodes = int(m + 1)

    # Optionally attach node embeddings if present and size matches 'entity'
    # E.g., clip_text_node_emb_en shaped [N_entity, D]
    node_emb = None
    for k in ["clip_text_node_emb_en", "node_emb", "node_embeddings"]:
        if k in payload:
            try:
                node_emb = torch.as_tensor(payload[k])
                break
            except Exception:
                continue
    if node_emb is not None and "entity" in data.node_types:
        try:
            n_entity = int(data["entity"].num_nodes)
            if node_emb.dim() == 1:
                node_emb = node_emb.view(-1, 1)
            if node_emb.size(0) == n_entity:
                data["entity"].x = node_emb.to(torch.float)
        except Exception:
            pass

    return data


def main() -> int:
    args = parse_args()
    payload = torch.load(args.inp, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("Input file must contain a dict payload including edge_index_label_{lang}.")
    hetero = build_hetero_from_label_edges(payload, args.lang)
    torch.save(hetero, args.out)
    print(f"Saved HeteroData with label-valued edge types to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


