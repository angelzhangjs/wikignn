import argparse
import os
import sys
import json
import torch
from typing import Any, Dict, Tuple

try:
    from torch_geometric.data import Data, HeteroData  # type: ignore
except Exception:
    Data = None
    HeteroData = None

def _shape_str(x: Any) -> str:
    try:
        if torch.is_tensor(x):
            return str(tuple(x.shape))
        if hasattr(x, 'shape') and x.shape is not None:
            return str(tuple(x.shape))
        if isinstance(x, (list, tuple)):
            return f"len={len(x)}"
        return f"type={type(x).__name__}"
    except Exception as e:
        return f"unavailable ({e})"

def summarize_loaded_dict(loaded: Dict[str, Any]) -> Dict[str, Any]:
    res = {}
    for key in [
        'node_index',
        'edge_index',
        'clip_text_node_emb_en',
        'clip_text_edge_mean_emb_en',
        'clip_text_rel_emb_en_by_pid',
        'clip_meta',
        'pyg_data',
        'y',
        'labels',
        'node_labels',
    ]:
        if key in loaded:
            val = loaded[key]
            if key == 'clip_text_rel_emb_en_by_pid' and isinstance(val, dict):
                # Show a sample item's shape if possible
                sample_shape = None
                dims_set = set()
                for _, v2 in val.items():
                    sample_shape = _shape_str(v2)
                    try:
                        tv = torch.as_tensor(v2)
                        if tv.dim() >= 1:
                            dims_set.add(int(tv.shape[-1]))
                    except Exception:
                        pass
                    if len(dims_set) >= 4:
                        break
                res[key] = {
                    "type": "dict",
                    "num_items": len(val),
                    "sample_value_shape": sample_shape,
                    "embedding_dim_values_seen": sorted(list(dims_set)),
                }
            elif key in ('clip_text_node_emb_en', 'clip_text_edge_mean_emb_en'):
                entry = {"shape": _shape_str(val)}
                try:
                    tv = torch.as_tensor(val)
                    if tv.dim() >= 2:
                        entry["embedding_dim"] = int(tv.shape[-1])
                    elif tv.dim() == 1:
                        entry["embedding_dim"] = int(tv.shape[0])
                except Exception:
                    pass
                res[key] = entry
            elif key == 'clip_meta':
                meta_summary = {"type": type(val).__name__}
                if isinstance(val, dict):
                    meta_summary["keys"] = list(val.keys())
                res[key] = meta_summary
            else:
                res[key] = {"shape": _shape_str(val)}
    res["all_keys"] = list(loaded.keys())
    return res


def summarize_hetero(data: "HeteroData") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    node_summaries = []
    for node_type in getattr(data, 'node_types', []):
        store = data[node_type]
        try:
            n = int(store.num_nodes)
        except Exception:
            x = getattr(store, 'x', None)
            n = int(x.size(0)) if x is not None else 0
        x = getattr(store, 'x', None)
        x_dim = int(x.size(1)) if (x is not None and x.dim() == 2) else None
        y = getattr(store, 'y', None)
        num_labels = int(y.numel()) if y is not None else 0
        node_summaries.append({
            "type": node_type,
            "num_nodes": n,
            "has_x": x is not None,
            "x_dim": x_dim,
            "has_y": y is not None,
            "num_labels": num_labels,
        })
    edge_summaries = []
    for (src, rel, dst) in getattr(data, 'edge_types', []):
        store = data[(src, rel, dst)]
        ei = getattr(store, 'edge_index', None)
        num_edges = int(ei.size(1)) if (ei is not None and ei.dim() == 2) else 0
        ea = getattr(store, 'edge_attr', None)
        edge_dim = int(ea.size(1)) if (ea is not None and ea.dim() == 2) else None
        edge_summaries.append({
            "edge_type": (src, rel, dst),
            "num_edges": num_edges,
            "has_edge_attr": ea is not None,
            "edge_attr_dim": edge_dim,
        })
    out["graph_type"] = "HeteroData"
    out["node_types"] = node_summaries
    out["edge_types"] = edge_summaries
    return out


def summarize_homo(data: "Data") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    n = int(data.num_nodes) if getattr(data, 'num_nodes', None) is not None else int(data.x.size(0))
    ei = getattr(data, 'edge_index', None)
    e = int(ei.size(1)) if (ei is not None and ei.dim() == 2) else 0
    x = getattr(data, 'x', None)
    x_dim = int(x.size(1)) if (x is not None and x.dim() == 2) else None
    ea = getattr(data, 'edge_attr', None)
    edge_dim = int(ea.size(1)) if (ea is not None and ea.dim() == 2) else None
    y = getattr(data, 'y', None)
    num_labels = int(y.numel()) if y is not None else 0
    out["graph_type"] = "Data"
    out["num_nodes"] = n
    out["num_edges"] = e
    out["has_x"] = x is not None
    out["x_dim"] = x_dim
    out["has_edge_attr"] = ea is not None
    out["edge_attr_dim"] = edge_dim
    out["has_y"] = y is not None
    out["num_labels"] = num_labels
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="graph_output/clean_graph.pyg_en_labeled.pt", help="Path to .pt file")
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    parser.add_argument("--out", type=str, default=None, help="Write JSON summary to this file path")
    args = parser.parse_args()

    if Data is None:
        print("ERROR: torch_geometric not available in this environment.")
        sys.exit(1)

    loaded = torch.load(args.path, map_location="cpu")
    # Determine data object if embedded
    if isinstance(loaded, dict) and 'pyg_data' in loaded:
        data = loaded['pyg_data']
    else:
        # It might already be Data/HeteroData, or a dict suitable for Data
        if isinstance(loaded, (Data, HeteroData)):
            data = loaded
        elif isinstance(loaded, dict):
            try:
                data = Data(**loaded)
            except Exception:
                data = loaded  # keep as dict
        else:
            data = loaded

    top_level = summarize_loaded_dict(loaded) if isinstance(loaded, dict) else {"raw_type": type(loaded).__name__}

    if HeteroData is not None and isinstance(data, HeteroData):
        core = summarize_hetero(data)
    elif isinstance(data, Data):
        core = summarize_homo(data)
    else:
        core = {"graph_type": type(data).__name__, "note": "Not a recognized PyG Data/HeteroData object"}

    summary = {"core": core, "top_level": top_level}
    # Persist to file if requested
    if args.out:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary JSON to {out_path}")
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("Graph core:")
        print(json.dumps(core, indent=2))
        print("\nTop-level container keys/shapes:")
        print(json.dumps(top_level, indent=2))


if __name__ == "__main__":
    main()


