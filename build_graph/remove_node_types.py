#!/usr/bin/env python3
"""
Remove specified node types (e.g., 'literal') from a PyG HeteroData graph.
All edges incident to the removed node types are dropped. Remaining node/edge
stores and their attributes (x, y, masks, edge_attr, etc.) are preserved.

Usage:
  python remove_node_types.py \
    --in graph_embedding/graph_by_label.pyg.pt.with_clip.pt \
    --out graph_embedding/graph_by_label.no_literal.pyg.pt \
    --drop literal
"""
from __future__ import annotations

import argparse
import sys
from typing import Iterable, Tuple

import torch
from torch_geometric.data import Data, HeteroData  # type: ignore


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Remove node types from a HeteroData graph")
    ap.add_argument("--in", dest="inp", required=True, help="Input .pt (HeteroData or dict with 'pyg_data')")
    ap.add_argument("--out", required=True, help="Output .pt path")
    ap.add_argument("--drop", nargs="+", default=["literal"], help="Node types to remove (default: literal)")
    return ap.parse_args()


def copy_node_store(dst: HeteroData, src: HeteroData, ntype: str) -> None:
    store = src[ntype]
    # Copy all attributes generically
    for key, value in store.items():
        # num_nodes can be copied as attribute as well
        setattr(dst[ntype], key, value)
    # Ensure num_nodes is set (in case it exists as a property only)
    num_nodes = getattr(store, "num_nodes", None)
    if num_nodes is not None:
        dst[ntype].num_nodes = int(num_nodes)


def copy_edge_store(dst: HeteroData, src: HeteroData, etype: Tuple[str, str, str]) -> None:
    store = src[etype]
    for key, value in store.items():
        setattr(dst[etype], key, value)


def remove_node_types(data: HeteroData, drop_types: Iterable[str]) -> HeteroData:
    drop_set = set(drop_types)
    out = HeteroData()
    # Copy node stores except dropped types
    for ntype in data.node_types:
        if ntype in drop_set:
            continue
        copy_node_store(out, data, ntype)
    # Copy edge stores that do not involve dropped node types
    for (src, rel, dst) in data.edge_types:
        if src in drop_set or dst in drop_set:
            continue
        copy_edge_store(out, data, (src, rel, dst))
    return out


def load_graph(path: str):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "pyg_data" in obj:
        return obj["pyg_data"]
    return obj


def save_graph(graph, path: str) -> None:
    torch.save(graph, path)


def main() -> int:
    args = parse_args()
    graph = load_graph(args.inp)
    if not isinstance(graph, (HeteroData, Data)):
        print("ERROR: Input file must contain a PyG Data/HeteroData object or dict with 'pyg_data'.", file=sys.stderr)
        return 2
    if isinstance(graph, Data):
        # Homogeneous graphs have no node types to drop
        print("Input graph is homogeneous; nothing to remove.", file=sys.stderr)
        save_graph(graph, args.out)
        print(f"Saved graph to {args.out}")
        return 0
    cleaned = remove_node_types(graph, args.drop)
    save_graph(cleaned, args.out)
    print(f"Dropped node types {args.drop} and saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


