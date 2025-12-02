#!/usr/bin/env python3
import argparse
from typing import Dict, List, Tuple, Any

import torch

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Print summary stats for a labeled graph payload (.pt)")
    ap.add_argument("--pt", required=True, help="Path to labeled .pt (e.g., clean_graph.pyg_en_labeled.pt)")
    ap.add_argument("--lang", default="en", help="Language suffix to inspect (default: en)")
    ap.add_argument("--top_k", type=int, default=20, help="Top-K relations to print (use --all to print all)")
    ap.add_argument("--all", dest="print_all", action="store_true", help="Print all relations (may be long)")
    return ap.parse_args()


def format_num(n: int) -> str:
    return f"{n:,}"


def sum_edges(edge_index: Dict[str, torch.Tensor]) -> int:
    total = 0
    for t in edge_index.values():
        try:
            total += int(t.shape[1])
        except Exception:
            pass
    return total


def relation_counts(edge_index: Dict[str, torch.Tensor]) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    for k, t in edge_index.items():
        try:
            items.append((str(k), int(t.shape[1])))
        except Exception:
            continue
    items.sort(key=lambda kv: kv[1], reverse=True)
    return items


def main() -> int:
    args = parse_args()
    p = torch.load(args.pt, map_location="cpu")
    if not isinstance(p, dict):
        raise TypeError("Expected a dict payload (.pt).")

    print(f"File: {args.pt}")
    print(f"Keys: {sorted(p.keys())}")

    # Node index overview
    node_index: Dict[str, Dict[str, int]] = p.get("node_index", {})
    if isinstance(node_index, dict):
        print("\nNode types (node_index):")
        total_nodes = 0
        for ntype, id2idx in node_index.items():
            n = len(id2idx) if isinstance(id2idx, dict) else 0
            total_nodes += n
            print(f"  {ntype}: {format_num(n)}")
        print(f"  Total nodes: {format_num(total_nodes)}")
    else:
        print("\nNo node_index found.")

    # Edge overview (by PID relation key)
    edge_index: Dict[str, torch.Tensor] = p.get("edge_index", {})
    if isinstance(edge_index, dict) and edge_index:
        total_edges = sum_edges(edge_index)
        print(f"\nEdges (by relation key srcType:Pxx:dstType): total={format_num(total_edges)}")
        rel_items = relation_counts(edge_index)
        if args.print_all:
            for k, cnt in rel_items:
                print(f"  {k}: {format_num(cnt)}")
        else:
            top_k = max(0, args.top_k)
            for k, cnt in rel_items[:top_k]:
                print(f"  {k}: {format_num(cnt)}")
            if len(rel_items) > top_k:
                print(f"  ... ({len(rel_items) - top_k} more)")
    else:
        print("\nNo edge_index found.")

    # Claims and qualifiers coverage
    edge_claim_ids: Dict[str, List[Any]] = p.get("edge_claim_ids", {}) or {}
    num_claim_lists = 0
    num_claims_total = 0
    if isinstance(edge_claim_ids, dict):
        for k, claim_list in edge_claim_ids.items():
            if isinstance(claim_list, list):
                num_claim_lists += 1
                num_claims_total += len(claim_list)
    print(f"\nClaims: lists={format_num(num_claim_lists)} | total_claim_ids={format_num(num_claims_total)}")

    qualifiers_by_claim_id: Dict[str, Dict[str, Any]] = p.get("qualifiers_by_claim_id", {}) or {}
    num_claim_nodes = len(qualifiers_by_claim_id) if isinstance(qualifiers_by_claim_id, dict) else 0
    print(f"Qualifiers: claim_nodes={format_num(num_claim_nodes)}")

    # Labels and labeled regroupings
    prop_key = f"property_labels_{args.lang}"
    rel_key = f"relation_labels_{args.lang}"
    ei_label_key = f"edge_index_label_{args.lang}"
    eci_label_key = f"edge_claim_ids_label_{args.lang}"
    et_pairs_key = f"edge_text_pairs_label_{args.lang}"

    if prop_key in p and isinstance(p[prop_key], dict):
        print(f"\nProperty labels ({args.lang}): {format_num(len(p[prop_key]))}")
    else:
        print(f"\nProperty labels ({args.lang}): none")

    if rel_key in p and isinstance(p[rel_key], dict):
        print(f"Relation labels ({args.lang}): {format_num(len(p[rel_key]))}")
    else:
        print(f"Relation labels ({args.lang}): none")

    if ei_label_key in p and isinstance(p[ei_label_key], dict) and p[ei_label_key]:
        total_edges_lab = sum_edges(p[ei_label_key])
        print(f"Labeled edges (by relation label): total={format_num(total_edges_lab)}")
        rel_lab_items = relation_counts(p[ei_label_key])
        if args.print_all:
            for k, cnt in rel_lab_items:
                print(f"  {k}: {format_num(cnt)}")
        else:
            top_k = max(0, args.top_k)
            for k, cnt in rel_lab_items[:top_k]:
                print(f"  {k}: {format_num(cnt)}")
            if len(rel_lab_items) > top_k:
                print(f"  ... ({len(rel_lab_items) - top_k} more)")
    else:
        print("Labeled edges: none")

    if eci_label_key in p and isinstance(p[eci_label_key], dict):
        num_rel_with_claims = sum(1 for v in p[eci_label_key].values() if isinstance(v, list) and v)
        print(f"Labeled claim lists: relations_with_claims={format_num(num_rel_with_claims)}")
    else:
        print("Labeled claim lists: none")

    # Entity metadata presence
    lbl_ent_key = f"entity_labels_{args.lang}"
    desc_ent_key = f"entity_descriptions_{args.lang}"
    alias_ent_key = f"entity_aliases_{args.lang}"
    text_by_idx_key = f"entity_text_{args.lang}_by_index"

    def _len_if_dict(d):
        return len(d) if isinstance(d, dict) else 0

    print("\nEntity metadata:")
    print(f"  labels: {format_num(_len_if_dict(p.get(lbl_ent_key)))}")
    print(f"  descriptions: {format_num(_len_if_dict(p.get(desc_ent_key)))}")
    print(f"  aliases: {format_num(_len_if_dict(p.get(alias_ent_key)))}")
    print(f"  text_by_index: {format_num(_len_if_dict(p.get(text_by_idx_key)))}")

    # Text pairs (if available)
    if et_pairs_key in p and isinstance(p[et_pairs_key], dict) and p[et_pairs_key]:
        total_pairs = 0
        per_rel = []
        for k, pairs in p[et_pairs_key].items():
            n = len(pairs) if isinstance(pairs, list) else 0
            total_pairs += n
            per_rel.append((str(k), n))
        per_rel.sort(key=lambda kv: kv[1], reverse=True)
        print(f"\nEdge text pairs ({args.lang}): total_pairs={format_num(total_pairs)} relations={format_num(len(per_rel))}")
        if args.print_all:
            for k, n in per_rel:
                print(f"  {k}: {format_num(n)}")
        else:
            top_k = max(0, args.top_k)
            for k, n in per_rel[:top_k]:
                print(f"  {k}: {format_num(n)}")
            if len(per_rel) > top_k:
                print(f"  ... ({len(per_rel) - top_k} more)")
    else:
        print("\nEdge text pairs: none")

    # Meta
    meta = p.get("meta")
    if isinstance(meta, dict):
        print("\nMeta keys:", sorted(meta.keys()))

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


