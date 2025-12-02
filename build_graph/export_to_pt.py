#!/usr/bin/env python3
"""
Convert an artwork graph directory (nodes/edges JSONL) to a .pt file.

Outputs a torch-saved payload with:
  - node indices per node type ('entity', 'literal')
  - edge_index tensors per (src_type, relation/property_id, dst_type)
  - edge claim_ids aligned with edges for traceability
  - qualifiers_by_claim_id for downstream feature building

Optionally, if torch_geometric is installed and --format pyg is provided,
also save a torch_geometric.data.HeteroData graph (under key 'pyg_data').
"""
from __future__ import annotations
import argparse
import json
import os
from typing import Dict, Iterator, List, Tuple, Optional

import torch
try:
    from torch_geometric.data import HeteroData  # type: ignore
    HAS_PYG = True
except Exception:
    HAS_PYG = False

def iter_jsonl(path: str) -> Iterator[dict]:
    if not path or not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def build_node_indices(graph_dir: str) -> Dict[str, Dict[str, int]]:
    """
    Build indices for nodes from nodes.jsonl for 'entity' nodes.
    Literal nodes will be added on the fly when parsing edges.
    """
    node_index: Dict[str, Dict[str, int]] = {
        "entity": {},
        "literal": {},
    }
    nodes_path = os.path.join(graph_dir, "nodes.jsonl")
    next_idx = 0
    for row in iter_jsonl(nodes_path):
        qid = row.get("qid")
        if not isinstance(qid, str):
            continue
        if qid not in node_index["entity"]:
            node_index["entity"][qid] = next_idx
            next_idx += 1
    return node_index

def get_or_add(node_index: Dict[str, Dict[str, int]], node_type: str, node_id: str, next_idx_ref: Dict[str, int]) -> int:
    """
    Fetch node index or add a new index for the given node_type/node_id.
    next_idx_ref holds the next index per node_type.
    """
    idx_map = node_index[node_type]
    if node_id in idx_map:
        return idx_map[node_id]
    idx = next_idx_ref[node_type]
    idx_map[node_id] = idx
    next_idx_ref[node_type] = idx + 1
    return idx

def export_graph_to_pt(graph_dir: str, out_path: str, out_format: str = "basic") -> None:
    """
    Convert graph JSONLs to a torch-saved .pt file with adjacency and metadata.
    out_format: 'basic' (always works) or 'pyg' (requires torch_geometric).
    """
    assert out_format in ("basic", "pyg")

    # 1) Build node indices for entities; init counters for other types
    node_index = build_node_indices(graph_dir)
    next_idx = {
        "entity": (max(node_index["entity"].values()) + 1) if node_index["entity"] else 0,
        "literal": 0,
    }

    # 2) Prepare edge collectors
    # Map: (src_type, relation/property_id, dst_type) -> lists of (src_idx, dst_idx) and claim_ids
    edge_src: Dict[Tuple[str, str, str], List[int]] = {}
    edge_dst: Dict[Tuple[str, str, str], List[int]] = {}
    edge_claims: Dict[Tuple[str, str, str], List[Optional[str]]] = {}

    def append_edge(key: Tuple[str, str, str], s: int, d: int, claim_id: Optional[str]) -> None:
        edge_src.setdefault(key, []).append(s)
        edge_dst.setdefault(key, []).append(d)
        edge_claims.setdefault(key, []).append(claim_id)

    # 3) Q->Q edges: entity_rels (outgoing + incoming if present)
    for fname in ("edges_entity_rels_outgoing.jsonl", "edges_entity_rels_incoming.jsonl"):
        path = os.path.join(graph_dir, fname)
        for row in iter_jsonl(path):
            prop = row.get("property_id")
            if not isinstance(prop, str):
                continue
            src_q = row.get("source")
            dst_q = row.get("target")
            if not (isinstance(src_q, str) and isinstance(dst_q, str)):
                continue
            s_idx = get_or_add(node_index, "entity", src_q, next_idx)
            d_idx = get_or_add(node_index, "entity", dst_q, next_idx)
            key = ("entity", prop, "entity")
            append_edge(key, s_idx, d_idx, row.get("claim_id"))

    # 4) Q->literal edges: entity_values
    for row in iter_jsonl(os.path.join(graph_dir, "edges_entity_values.jsonl")):
        prop = row.get("property_id")
        if not isinstance(prop, str):
            continue
        src_q = row.get("source")
        value = row.get("value")
        if not (isinstance(src_q, str) and isinstance(value, str)):
            continue
        lit_id = f"lit:{prop}:{value}"
        s_idx = get_or_add(node_index, "entity", src_q, next_idx)
        d_idx = get_or_add(node_index, "literal", lit_id, next_idx)
        key = ("entity", prop, "literal")
        append_edge(key, s_idx, d_idx, row.get("claim_id"))
        
    # # 5) Q->external_id edges: external_ids
    # for row in iter_jsonl(os.path.join(graph_dir, "edges_external_ids.jsonl")):
    #     prop = row.get("property_id")
    #     if not isinstance(prop, str):
    #         continue
    #     src_q = row.get("source")
    #     ext_val = row.get("value")
    #     if not (isinstance(src_q, str) and isinstance(ext_val, str)):
    #         continue
    #     ext_id = f"ext:{prop}:{ext_val}"
    #     s_idx = get_or_add(node_index, "entity", src_q, next_idx)
    #     d_idx = get_or_add(node_index, "external_id", ext_id, next_idx)
    #     key = ("entity", prop, "external_id")
    #     append_edge(key, s_idx, d_idx, row.get("claim_id"))

    # 6) Qualifiers map by claim_id
    qualifiers_by_claim_id: Dict[str, List[dict]] = {}
    for row in iter_jsonl(os.path.join(graph_dir, "qualifiers.jsonl")):
        claim_id = row.get("claim_id")
        if not isinstance(claim_id, str):
            continue
        qualifiers_by_claim_id.setdefault(claim_id, []).append({
            "qualifier_id": row.get("qualifier_id"),
            "property_id": row.get("property_id"),
            "value": row.get("value"),
        })

    # 7) Convert to tensors
    edge_index: Dict[str, torch.Tensor] = {}
    edge_claim_ids: Dict[str, List[Optional[str]]] = {}
    for (src_t, rel, dst_t), s_list in edge_src.items():
        d_list = edge_dst[(src_t, rel, dst_t)]
        key_str = f"{src_t}:{rel}:{dst_t}"
        edge_index[key_str] = torch.tensor([s_list, d_list], dtype=torch.long)
        edge_claim_ids[key_str] = edge_claims[(src_t, rel, dst_t)]

    # 8) Package payload
    payload: Dict[str, object] = {
        "node_index": node_index,  # dict of dicts: type -> id -> idx
        "edge_index": edge_index,  # dict: "src:rel:dst" -> LongTensor [2, E]
        "edge_claim_ids": edge_claim_ids,  # dict: "src:rel:dst" -> List[str|None]
        "qualifiers_by_claim_id": qualifiers_by_claim_id,
        "meta": {
            "num_nodes": {k: len(v) for k, v in node_index.items()},
            "num_edges": {k: int(v.size(1)) for k, v in edge_index.items()},
        },
    }

    # 9) Optional PyG graph
    if out_format == "pyg":
        if not HAS_PYG:
            raise RuntimeError("torch_geometric not available; install PyG or use --format basic")
        data = HeteroData()
        # set node counts
        for ntype, indices in node_index.items():
            # Creating empty x allows downstream code to know node count
            data[ntype].num_nodes = len(indices)
            print(f"Node type: {ntype}, count: {len(indices)}")
        # add edges per relation
        for key_str, ei in edge_index.items():
            src_t, rel, dst_t = key_str.split(":")
            data[(src_t, rel, dst_t)].edge_index = ei
        payload["pyg_data"] = data

    # 10) Save
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(payload, out_path)
    print(f"Saved: {out_path}")
    print(f"Nodes: {payload['meta']['num_nodes']}")
    print("Edges: {{k: v.shape[1] for k, v in edge_index.items()}}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export artwork graph JSONL to .pt")
    ap.add_argument("--graph_dir", required=True, help="Path to graph_out_dir with nodes/edges JSONL")
    ap.add_argument("--out", required=True, help="Output .pt path")
    ap.add_argument("--format", choices=["basic", "pyg"], default="basic", help="Output payload format")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    export_graph_to_pt(args.graph_dir, args.out, args.format)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


