#!/usr/bin/env python3
import argparse
import os
import math
from typing import Dict, Tuple, List, Optional, Any

import torch


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Filter a .pt graph to remove null/invalid data.")
    ap.add_argument("--input", "-i", required=True, help="Path to input graph .pt (e.g., graph.pyg.pt)")
    ap.add_argument("--output", "-o", default=None, help="Path to save cleaned .pt (default: alongside input with _non_null suffix)")
    ap.add_argument("--drop_empty_relations", action="store_true", help="Drop relation keys that end up with 0 edges after filtering")
    ap.add_argument("--drop_null_claims", action="store_true", help="Remove edges with null/empty claim_id")
    ap.add_argument("--dedup_edges", action="store_true", help="Remove duplicate (src,dst,relation) edges (keeps first)")
    ap.add_argument("--enforce_index_bounds", action="store_true", help="Drop edges with indices >= declared num_nodes_by_type when node_index is missing")
    return ap.parse_args()


def is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in ("", "null", "none", "nan"):
        return True
    return False


def as_two_lists(ei: Any) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    if ei is None:
        return None, None
    try:
        if hasattr(ei, "tolist"):
            src = ei[0].tolist()
            dst = ei[1].tolist()
        else:
            src = list(ei[0])
            dst = list(ei[1])
        return src, dst
    except Exception:
        return None, None


def build_allowed_index_sets(node_index: Optional[Dict[str, Dict[str, int]]]) -> Dict[str, set]:
    allowed: Dict[str, set] = {}
    if not isinstance(node_index, dict):
        return allowed
    for ntype, id_to_idx in node_index.items():
        try:
            idxs = set(int(v) for v in id_to_idx.values())
        except Exception:
            idxs = set()
        allowed[ntype] = idxs
    return allowed


def build_bounds_by_type(payload: Dict[str, Any]) -> Dict[str, int]:
    """
    Returns exclusive upper bounds for indices per node type if available.
    Recognized keys:
      - num_nodes_by_type: {node_type: int}
      - num_nodes: int (applied if only one node type is present overall)
    """
    bounds: Dict[str, int] = {}
    num_nodes_by_type = payload.get("num_nodes_by_type")
    if isinstance(num_nodes_by_type, dict):
        for ntype, n in num_nodes_by_type.items():
            try:
                bounds[str(ntype)] = int(n)
            except Exception:
                continue
    else:
        # Fallback: if only a single node type exists across relations, use num_nodes if present
        edge_index = payload.get("edge_index")
        if isinstance(edge_index, dict):
            types = set()
            for key_str in edge_index.keys():
                try:
                    src_t, _prop, dst_t = key_str.split(":")
                except Exception:
                    continue
                types.add(src_t)
                types.add(dst_t)
            if len(types) == 1 and "num_nodes" in payload:
                try:
                    bounds[list(types)[0]] = int(payload["num_nodes"])
                except Exception:
                    pass
    return bounds


def filter_payload_dict(
    payload: Dict[str, Any],
    drop_empty_relations: bool,
    drop_null_claims: bool = False,
    dedup_edges: bool = False,
    enforce_index_bounds: bool = False,
) -> Dict[str, Any]:
    """
    Expect shape (flexible, we only depend on a few keys):
      - node_index: {node_type: {id: idx}}
      - edge_index: { "srcType:prop:dstType": LongTensor shape [2, E] }
      - edge_claim_ids: { relation_key: [str|None]*E }  (optional)
    """
    node_index = payload.get("node_index")
    edge_index = payload.get("edge_index")
    edge_claim_ids = payload.get("edge_claim_ids", {})

    if not isinstance(edge_index, dict):
        raise TypeError("Input .pt does not contain expected 'edge_index' dict payload")

    allowed_by_type = build_allowed_index_sets(node_index if isinstance(node_index, dict) else None)
    bounds_by_type = build_bounds_by_type(payload) if enforce_index_bounds and not allowed_by_type else {}

    cleaned_edge_index: Dict[str, torch.Tensor] = {}
    cleaned_edge_claim_ids: Dict[str, List[Optional[str]]] = {}

    before_total = 0
    after_total = 0

    for key_str, ei in edge_index.items():
        before_src, before_dst = as_two_lists(ei)
        if before_src is None or before_dst is None:
            continue
        before_E = min(len(before_src), len(before_dst))
        before_total += before_E

        try:
            src_t, _prop, dst_t = key_str.split(":")
        except Exception:
            src_t, _prop, dst_t = "unknown", str(key_str), "unknown"

        allowed_src = allowed_by_type.get(src_t)
        allowed_dst = allowed_by_type.get(dst_t)
        bound_src = bounds_by_type.get(src_t)
        bound_dst = bounds_by_type.get(dst_t)

        claims_list = edge_claim_ids.get(key_str)
        if claims_list is not None and len(claims_list) != before_E:
            # Length mismatch; ignore claims for this relation to avoid misalignment
            claims_list = None

        new_src: List[int] = []
        new_dst: List[int] = []
        new_claims: List[Optional[str]] = []
        seen_pairs = set()

        for i in range(before_E):
            s_val = before_src[i]
            d_val = before_dst[i]
            try:
                si = int(s_val) if not is_nullish(s_val) else None
                di = int(d_val) if not is_nullish(d_val) else None
            except Exception:
                si, di = None, None

            if si is None or di is None:
                continue
            if si < 0 or di < 0:
                continue
            if allowed_src is not None and si not in allowed_src:
                continue
            if allowed_dst is not None and di not in allowed_dst:
                continue
            if enforce_index_bounds and not allowed_by_type:
                if bound_src is not None and si >= bound_src:
                    continue
                if bound_dst is not None and di >= bound_dst:
                    continue

            if dedup_edges:
                key_pair = (si, di)
                if key_pair in seen_pairs:
                    continue
                seen_pairs.add(key_pair)

            # Keep edge
            if claims_list is not None:
                claim_val = claims_list[i]
                if drop_null_claims and is_nullish(claim_val):
                    continue
                new_claims.append(None if is_nullish(claim_val) else claim_val)
            new_src.append(si)
            new_dst.append(di)

        if len(new_src) == 0 and drop_empty_relations:
            continue

        cleaned_edge_index[key_str] = torch.tensor([new_src, new_dst], dtype=torch.long)
        after_total += len(new_src)
        if new_claims:
            cleaned_edge_claim_ids[key_str] = new_claims

    cleaned: Dict[str, Any] = {}

    # Keep original node_index if present
    if isinstance(node_index, dict):
        cleaned["node_index"] = node_index
    cleaned["edge_index"] = cleaned_edge_index
    if cleaned_edge_claim_ids:
        cleaned["edge_claim_ids"] = cleaned_edge_claim_ids

    # Pass-through other top-level metadata keys unchanged (non-destructive)
    for k, v in payload.items():
        if k in ("node_index", "edge_index", "edge_claim_ids"):
            continue
        cleaned[k] = v

    print(f"Filtered edges: before={before_total}, after={after_total}, removed={before_total - after_total}")
    print(f"Relations kept: {len(cleaned_edge_index)}")
    return cleaned


def main() -> int:
    args = parse_args()
    in_path = args.input
    if args.output:
        out_path = args.output
    else:
        # Default to a canonical cleaned filename alongside the input
        out_path = os.path.join(os.path.dirname(in_path), "clean_graph.pyg.pt")

    obj = torch.load(in_path, map_location="cpu")

    if isinstance(obj, dict) and "edge_index" in obj:
        cleaned = filter_payload_dict(
            obj,
            drop_empty_relations=args.drop_empty_relations,
            drop_null_claims=args.drop_null_claims,
            dedup_edges=args.dedup_edges,
            enforce_index_bounds=args.enforce_index_bounds,
        )
        torch.save(cleaned, out_path)
        print(f"Saved cleaned payload to: {out_path}")
        return 0

    # If it's a PyG Data/HeteroData, we conservatively keep structure and only
    # drop edges with NaNs in edge_attr (if present). We avoid node drops/remap.
    try:
        from torch_geometric.data import Data, HeteroData  # type: ignore
    except Exception:
        Data = None  # type: ignore
        HeteroData = None  # type: ignore

    if Data is not None and isinstance(obj, Data):
        data = obj.clone()
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            ea = data.edge_attr
            if torch.is_floating_point(ea):
                mask = ~torch.isnan(ea).any(dim=1)
                data.edge_index = data.edge_index[:, mask]
                data.edge_attr = ea[mask]
            else:
                # Non-floating attrs -> keep as-is (no NaNs)
                pass
        torch.save(data, out_path)
        print(f"Saved cleaned Data to: {out_path}")
        return 0

    if HeteroData is not None and isinstance(obj, HeteroData):
        data = obj.clone()
        # Iterate edge types; drop edges with NaNs in edge attributes if present
        for edge_type in list(data.edge_types):
            store = data[edge_type]
            if "edge_attr" in store and store["edge_attr"] is not None:
                ea = store["edge_attr"]
                if torch.is_floating_point(ea):
                    mask = ~torch.isnan(ea).any(dim=1)
                    store["edge_index"] = store["edge_index"][:, mask]
                    store["edge_attr"] = ea[mask]
        torch.save(data, out_path)
        print(f"Saved cleaned HeteroData to: {out_path}")
        return 0

    # Fallback: Not a supported format
    raise TypeError(f"Unsupported input object type: {type(obj)}")


if __name__ == "__main__":
    raise SystemExit(main())


