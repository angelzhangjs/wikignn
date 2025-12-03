#!/usr/bin/env python3
"""
Create a weakly supervised retrieval dataset from a hetero graph for a given query.
- Select top-N positive entity seeds by SBERT similarity to the query text
- Select N negative seeds from the lowest-similarity entities
- Extract k-hop subgraphs around each seed (entity node type)
- Save subgraphs as .pt and write a JSONL pairs file with labels {1,0}

Example:
  python make_pairs_from_query.py \
    --graph graph_embedding/graph_by_label.no_literal.pyg.pt \
    --out_dir graph_embedding/pairs_sample \
    --query "Renaissance Italian painter and polymath" \
    --num_pos 500 --num_neg 500 --hops 2
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple, Optional

import torch
from torch_geometric.data import HeteroData  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build (query, subgraph) pairs with weak labels")
    ap.add_argument("--graph", required=True, help="Path to HeteroData .pt (or dict with 'pyg_data')")
    ap.add_argument("--out_dir", default="query_graph", help="Output directory for subgraphs and pairs.jsonl (default: query_graph)")
    ap.add_argument("--query", required=True, help="Query text to retrieve for")
    ap.add_argument("--num_pos", type=int, default=500, help="Number of positive subgraphs")
    ap.add_argument("--num_neg", type=int, default=500, help="Number of negative subgraphs")
    ap.add_argument("--hops", type=int, default=2, help="Number of hops for subgraphs (1-2 recommended)")
    ap.add_argument("--sbert_model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Sentence-Transformers model name")
    ap.add_argument("--max_chars", type=int, default=300, help="Cap entity text length")
    return ap.parse_args()


def load_graph(path: str) -> Tuple[HeteroData, Dict]:
    obj = torch.load(path, map_location="cpu")
    meta: Dict = obj if isinstance(obj, dict) else {}
    data = meta.get("pyg_data", obj)
    if not isinstance(data, HeteroData):
        raise TypeError("Expected a HeteroData object or dict with 'pyg_data'.")
    return data, meta


def get_entity_texts(meta: Dict, num_nodes: int, max_chars: int) -> List[str]:
    # Prefer rich text; fallback to labels; last resort use node_index.entity inversion
    if "entity_text_en_by_index" in meta and isinstance(meta["entity_text_en_by_index"], dict):
        src = meta["entity_text_en_by_index"]
        texts = [""] * num_nodes
        for k, v in src.items():
            try:
                i = int(k)
                if 0 <= i < num_nodes:
                    s = str(v).replace("\n", " ").strip()
                    if max_chars and len(s) > max_chars:
                        s = s[:max_chars]
                    texts[i] = s
            except Exception:
                continue
        if any(texts):
            return texts
    if "entity_labels_en" in meta and isinstance(meta["entity_labels_en"], dict):
        src = meta["entity_labels_en"]
        texts = [""] * num_nodes
        for k, v in src.items():
            try:
                i = int(k)
                if 0 <= i < num_nodes:
                    s = str(v).replace("\n", " ").strip()
                    if max_chars and len(s) > max_chars:
                        s = s[:max_chars]
                    texts[i] = s
            except Exception:
                continue
        if any(texts):
            return texts
    # Invert node_index.entity if present
    node_index = meta.get("node_index")
    if isinstance(node_index, dict) and "entity" in node_index and isinstance(node_index["entity"], dict):
        mapping = node_index["entity"]
        texts = [""] * num_nodes
        for text_value, idx in mapping.items():
            try:
                i = int(idx)
                if 0 <= i < num_nodes:
                    s = str(text_value).replace("\n", " ").strip()
                    if max_chars and len(s) > max_chars:
                        s = s[:max_chars]
                    texts[i] = s
            except Exception:
                continue
        return texts
    # Fallback: empty strings
    return [""] * num_nodes


def sbert_encode(texts: List[str], model_name: str) -> torch.Tensor:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")
    model = SentenceTransformer(model_name, device="cpu")
    embs = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True).float()
    return embs.cpu()


def rank_entities_by_query(texts: List[str], query: str, model_name: str) -> torch.Tensor:
    embs = sbert_encode([query] + texts, model_name)
    q = embs[0]  # [D]
    X = embs[1:]  # [N, D]
    sim = (X @ q)  # cosine since normalized
    return sim  # shape [N]


def build_entity_adjacency(data: HeteroData, num_nodes: int) -> List[List[int]]:
    # Treat edges as undirected for neighborhood collection
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    for (src_t, _, dst_t), store in data.edge_items():
        if src_t != "entity" or dst_t != "entity":
            continue
        ei = store.get("edge_index", None)
        if ei is None:
            continue
        rows, cols = ei[0].tolist(), ei[1].tolist()
        for u, v in zip(rows, cols):
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                adj[u].append(v)
                adj[v].append(u)
    return adj


def k_hop_nodes(adj: List[List[int]], seed: int, hops: int) -> List[int]:
    seen = {seed}
    frontier = {seed}
    for _ in range(max(0, hops)):
        new_frontier = set()
        for u in frontier:
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    new_frontier.add(v)
        frontier = new_frontier
        if not frontier:
            break
    return sorted(seen)


def subgraph_from_nodes(data: HeteroData, ntype: str, nodes: List[int]) -> HeteroData:
    node_set = set(nodes)
    remap = {old: i for i, old in enumerate(nodes)}
    out = HeteroData()
    # Node store
    src_store = data[ntype]
    for key, val in src_store.items():
        if torch.is_tensor(val) and val.size(0) == len(src_store.x) if hasattr(src_store, "x") else val.size(0) == src_store.num_nodes:
            out[ntype][key] = val[nodes]
        else:
            out[ntype][key] = val
    out[ntype].num_nodes = len(nodes)
    # Edge stores: filter to edges within node_set and remap indices
    for et, store in data.edge_items():
        if et[0] != ntype or et[2] != ntype:
            continue
        ei = store.get("edge_index", None)
        if ei is None:
            continue
        rows, cols = ei[0], ei[1]
        mask = rows.new_zeros(rows.size(0), dtype=torch.bool)
        # build mask
        for i in range(rows.size(0)):
            u = int(rows[i].item()); v = int(cols[i].item())
            if u in node_set and v in node_set:
                mask[i] = True
        sel = mask.nonzero(as_tuple=False).view(-1)
        if sel.numel() == 0:
            continue
        rows_sub = rows[sel].tolist()
        cols_sub = cols[sel].tolist()
        rows_remap = [remap[int(u)] for u in rows_sub]
        cols_remap = [remap[int(v)] for v in cols_sub]
        out[et].edge_index = torch.tensor([rows_remap, cols_remap], dtype=torch.long)
        # copy aligned attrs (edge_attr, y, masks) where length==E
        E = out[et].edge_index.size(1)
        for key, val in store.items():
            if key == "edge_index":
                continue
            if torch.is_tensor(val) and val.size(0) == rows.size(0):
                out[et][key] = val[sel]
            else:
                out[et][key] = val
    return out


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    pairs_path = os.path.join(args.out_dir, "pairs.jsonl")
    sub_dir = os.path.join(args.out_dir, "subgraphs")
    os.makedirs(sub_dir, exist_ok=True)

    data, meta = load_graph(args.graph)
    if "entity" not in data.node_types:
        raise ValueError("Expected an 'entity' node type in the graph.")
    num_nodes = int(data["entity"].num_nodes)
    texts = get_entity_texts(meta, num_nodes, args.max_chars)

    sim = rank_entities_by_query(texts, args.query, args.sbert_model)  # [N]
    # positives: top-K; negatives: bottom-K
    order = torch.argsort(sim, descending=True)
    pos_ids = order[: args.num_pos].tolist()
    neg_ids = order[-args.num_neg :].tolist()

    adj = build_entity_adjacency(data, num_nodes)

    # Write pairs
    n_written_pos = 0
    n_written_neg = 0
    with open(pairs_path, "w") as f:
        # positives
        for nid in pos_ids:
            nodes = k_hop_nodes(adj, nid, args.hops)
            sub = subgraph_from_nodes(data, "entity", nodes)
            out_path = os.path.join(sub_dir, f"entity_{nid}_pos.pt")
            torch.save(sub, out_path)
            rec = {"query": args.query, "subgraph_path": out_path, "label": 1}
            f.write(json.dumps(rec) + "\n")
            n_written_pos += 1
        # negatives
        for nid in neg_ids:
            nodes = k_hop_nodes(adj, nid, args.hops)
            sub = subgraph_from_nodes(data, "entity", nodes)
            out_path = os.path.join(sub_dir, f"entity_{nid}_neg.pt")
            torch.save(sub, out_path)
            rec = {"query": args.query, "subgraph_path": out_path, "label": 0}
            f.write(json.dumps(rec) + "\n")
            n_written_neg += 1

    print(f"Wrote {n_written_pos} positives and {n_written_neg} negatives to {args.out_dir}")
    print(f"Pairs file: {pairs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


