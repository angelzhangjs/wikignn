from retrieval.pairs_dataset import PairDataset, collate_pairs
from torch.utils.data import DataLoader
import torch


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="graph_embedding/pairs_all/train.jsonl", help="Path to pairs.jsonl")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    ds = PairDataset(args.pairs)
    if len(ds) == 0:
        print("Empty pairs file:", args.pairs)
        return
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pairs)
    queries, batch, y = next(iter(loader))

    print("Num pairs in file:", len(ds))
    print("Batch size:", len(queries))
    print("Node types:", batch.node_types)
    print("Edge types:", batch.edge_types)

    # Node feature checks
    for ntype in batch.node_types:
        has_x = hasattr(batch[ntype], "x")
        shape = tuple(batch[ntype].x.shape) if has_x else None
        print(f"Node {ntype}: has_x={has_x}, x_shape={shape}")

    # Edge feature checks
    for et in batch.edge_types:
        ei = batch[et].edge_index
        ei_shape = tuple(ei.shape) if isinstance(ei, torch.Tensor) else None
        has_ea = hasattr(batch[et], "edge_attr")
        ea_shape = tuple(batch[et].edge_attr.shape) if has_ea else None
        print(f"Edge {et}: edge_index_shape={ei_shape}, has_edge_attr={has_ea}, edge_attr_shape={ea_shape}")

    print("Labels y shape:", tuple(y.shape), "dtype:", y.dtype)
    print("Sample queries (first 3):", queries[:3])

if __name__ == "__main__":
    main()