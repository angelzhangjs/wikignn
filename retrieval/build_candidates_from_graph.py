#!/usr/bin/env python3
from __future__ import annotations
"""
Build a directory of candidate subgraphs from a large HeteroData graph for retrieval.

Given a .pt that contains either:
  - a dict with key 'pyg_data' holding torch_geometric.data.HeteroData, or
  - a HeteroData saved directly,
this script samples k-hop neighborhoods around 'entity' nodes and saves each subgraph
to out_dir as an individual .pt file compatible with PairDataset._load_subgraph.
"""
import argparse
import os
from typing import Dict, Iterable, Tuple

import torch

try:
    from torch_geometric.data import HeteroData  # type: ignore
    from torch_geometric.loader import NeighborLoader  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires torch_geometric. Please install PyG.") from e

def _load_hetero_graph(path: str) -> HeteroData:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "pyg_data" in obj:
        obj = obj["pyg_data"]
    if not isinstance(obj, HeteroData):
        raise TypeError(f"Expected HeteroData or dict with 'pyg_data' at {path}, got {type(obj)}")
    return obj

def build_candidates(
    graph_pt: str,
    out_dir: str,
    seed_node_type: str = "entity",
    num_hops: int = 2,
    num_neighbors: int = 16,
    max_candidates: int | None = None,
    batch_size: int = 1,
) -> Tuple[int, Dict[str, int]]:
    data = _load_hetero_graph(graph_pt)
    if seed_node_type not in data.node_types:
        raise ValueError(f"Seed node type '{seed_node_type}' not found in graph. Available: {data.node_types}")

    os.makedirs(out_dir, exist_ok=True)

    # num_neighbors can be a list [per hop]; keep same per hop
    nn_list = [num_neighbors for _ in range(num_hops)]
    seeds = torch.arange(int(data[seed_node_type].num_nodes))
    if max_candidates is not None:
        seeds = seeds[: max_candidates]

    loader = NeighborLoader(
        data,
        num_neighbors=nn_list,
        input_nodes=(seed_node_type, seeds),
        batch_size=batch_size,
        shuffle=False,
    )

    saved = 0
    stats: Dict[str, int] = {"total": 0}
    for batch in loader:
        # NeighborLoader can return mini-batches with multiple seeds; split if needed
        # We save one subgraph per seed for consistency with training format.
        # Each seed is indicated by a 'batch' vector per node type; we split by unique values.
        # For small batch_size=1 this is already per-seed.
        # To keep implementation simple and robust, enforce batch_size=1 here.
        # Users can increase throughput by parallelizing multiple processes.
        if batch_size != 1:
            raise ValueError("Please run with --batch-size 1 to save one subgraph per file.")
        subg = batch
        # Determine the global seed id and local index within this subgraph
        try:
            seed_global_id = int(seeds[saved].item())
        except Exception:
            seed_global_id = int(saved)
        seed_local_index = -1
        try:
            n_id = subg[seed_node_type].n_id
            if n_id is not None:
                # n_id is a tensor mapping local -> global
                where = (n_id == seed_global_id).nonzero(as_tuple=False)
                if where.numel() > 0:
                    seed_local_index = int(where.view(-1)[0].item())
        except Exception:
            pass

        out_path = os.path.join(out_dir, f"cand_{saved:08d}.pt")
        torch.save({
            "pyg_data": subg,
            "seed_node_type": seed_node_type,
            "seed_global_id": seed_global_id,
            "seed_local_index": seed_local_index,
        }, out_path)
        saved += 1
        if saved % 100 == 0:
            print(f"Saved {saved} candidates...")
    stats["total"] = saved
    return saved, stats

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build candidate subgraphs for retrieval")
    ap.add_argument("--graph-pt", required=True, help="Path to large graph .pt (with 'pyg_data' or HeteroData)")
    ap.add_argument("--out-dir", required=True, help="Directory to save candidate subgraphs")
    ap.add_argument("--seed-node-type", default="entity", help="Seed node type to sample around (default: entity)")
    ap.add_argument("--num-hops", type=int, default=2, help="Number of hops to sample (default: 2)")
    ap.add_argument("--num-neighbors", type=int, default=16, help="Neighbors per hop per edge-type (default: 16)")
    ap.add_argument("--max-candidates", type=int, default=10000, help="Max number of seeds to export (default: 10k)")
    ap.add_argument("--batch-size", type=int, default=1, help="NeighborLoader batch size (require 1 for saving)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    total, stats = build_candidates(
        graph_pt=args.graph_pt,
        out_dir=args.out_dir,
        seed_node_type=args.seed_node_type,
        num_hops=args.num_hops,
        num_neighbors=args.num_neighbors,
        max_candidates=args.max_candidates,
        batch_size=args.batch_size,
    )
    print(f"Done. Saved {total} candidate subgraphs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


