#!/usr/bin/env python3
"""
Pair-level dataset and DataLoader utilities for (query, subgraph, label) retrieval training.

Each JSONL line in pairs.jsonl must look like:
  {"query": "some text", "subgraph_path": "path/to/subgraph.pt", "label": 1}

Usage example:
  from retrieval.pairs_dataset import PairDataset, collate_pairs
  ds = PairDataset("query_graph/pairs.jsonl")
  loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_pairs)
  for queries, hetero_batch, y in loader:
      ...
"""
from __future__ import annotations

import json
import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, HeteroData  # type: ignore


def _load_subgraph(path: str) -> HeteroData:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "pyg_data" in obj:
        obj = obj["pyg_data"]
    if not isinstance(obj, HeteroData):
        raise TypeError(f"Expected HeteroData at {path}, got {type(obj)}")
    return obj


class PairDataset(Dataset):
    def __init__(self, pairs_jsonl_path: str):
        self.pairs_path = pairs_jsonl_path
        self.records: List[dict] = []
        with open(pairs_jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[str, HeteroData, torch.Tensor]:
        rec = self.records[idx]
        query = rec["query"]
        subgraph_path = rec["subgraph_path"]
        label = float(rec.get("label", 1.0))
        subg = _load_subgraph(subgraph_path)
        y = torch.tensor(label, dtype=torch.float32)  # BCE labels {0.0,1.0}
        return query, subg, y


def collate_pairs(batch: List[Tuple[str, HeteroData, torch.Tensor]]):
    queries, subgraphs, labels = zip(*batch)
    hetero_batch = Batch.from_data_list(list(subgraphs))
    y = torch.stack(list(labels))  # [B], float
    return list(queries), hetero_batch, y


def make_loaders(
    train_jsonl: str,
    val_jsonl: str | None = None,
    test_jsonl: str | None = None,
    batch_size: int = 16,
    num_workers: int = 0,
):
    train_ds = PairDataset(train_jsonl)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_pairs
    )
    val_loader = None
    test_loader = None
    if val_jsonl:
        val_ds = PairDataset(val_jsonl)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_pairs
        )
    if test_jsonl:
        test_ds = PairDataset(test_jsonl)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_pairs
        )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="query_graph/pairs.jsonl", help="Path to pairs.jsonl")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    if not os.path.exists(args.pairs):
        raise SystemExit(f"Pairs file not found: {args.pairs}")
    ds = PairDataset(args.pairs)
    print("Num pairs:", len(ds))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pairs)
    queries, hetero_batch, y = next(iter(loader))
    print("Sample batch:")
    print("  queries:", len(queries))
    print("  hetero node types:", hetero_batch.node_types)
    print("  hetero edge types:", hetero_batch.edge_types)
    print("  labels shape:", tuple(y.shape))


