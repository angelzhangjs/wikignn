import argparse
import sys
import torch
import re

try:
    from torch_geometric.data import Data, HeteroData  # type: ignore
except Exception:
    Data = None
    HeteroData = None

try:
    import clip  # pip install git+https://github.com/openai/CLIP.git
except Exception:
    clip = None


def parse_edge_type(s: str):
    parts = tuple(s.split(","))
    if len(parts) != 3:
        raise ValueError("Invalid --edge-type format. Use src,rel,dst")
    return parts


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-12)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-12)
    return float((a * b).sum().item())


def _load_clip_model(model_name: str, device: str):
    model, _ = clip.load(model_name, device=device)
    model.eval()
    return model

def _clip_text_dim(model) -> int:
    with torch.no_grad():
        t = clip.tokenize(["x"]).to(next(model.parameters()).device)
        emb = model.encode_text(t)
    return int(emb.shape[-1])


def _ensure_model_with_dim(target_dim: int, requested_name: str | None, device: str):
    candidates = []
    if requested_name and requested_name.lower() != "auto":
        candidates.append(requested_name)
    candidates.extend(["ViT-B/32", "ViT-L/14", "RN50", "RN101", "RN50x16", "RN50x64", "ViT-B/16"])
    tried = []
    for name in candidates:
        try:
            model = _load_clip_model(name, device)
            dim = _clip_text_dim(model)
            tried.append((name, dim))
            if dim == target_dim:
                return model, name, dim
        except Exception:
            continue
    # Fallback: return first successful even if dim mismatches
    for name, dim in tried:
        try:
            model = _load_clip_model(name, device)
            return model, name, dim
        except Exception:
            continue
    raise RuntimeError("Could not load any CLIP model.")


def encode_texts_with_clip(texts, model=None, device=None, max_chars=512):
    if clip is None:
        raise RuntimeError("The 'clip' package is not installed. Install via: pip install git+https://github.com/openai/CLIP.git")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model, _ = clip.load("RN50", device=device)  # default
        model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), 512):
            batch = []
            for t in texts[i:i + 512]:
                s = str(t).replace("\n", " ").strip()
                if max_chars is not None and len(s) > max_chars:
                    s = s[:max_chars]
                batch.append(s)
            toks = clip.tokenize(batch, truncate=True).to(device)
            feats = model.encode_text(toks).float()
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
            outs.append(feats.cpu())
    return torch.cat(outs, dim=0)


def check_hetero(data: "HeteroData", container: dict, args: argparse.Namespace):
    if args.edge_type is not None:
        edge_types = [parse_edge_type(args.edge_type)]
    else:
        edge_types = list(getattr(data, "edge_types", []))
    rel_labels = container.get("relation_labels_en") or container.get("property_labels_en") or {}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for et in edge_types:
        store = data[et]
        edge_attr = getattr(store, "edge_attr", None)
        edge_index = getattr(store, "edge_index", None)
        if edge_index is None or edge_index.dim() != 2 or edge_index.size(1) == 0:
            print(f"{et}: no edges")
            continue
        if edge_attr is None or edge_attr.numel() == 0:
            print(f"{et}: edge_attr missing")
            continue
        if edge_attr.dim() != 2:
            print(f"{et}: edge_attr has unexpected shape {tuple(edge_attr.shape)}")
            continue
        e = int(edge_index.size(1))
        d = int(edge_attr.size(1))
        try:
            uniq = int(torch.unique(edge_attr, dim=0).size(0))
        except Exception:
            uniq = -1
        print(f"{et}: num_edges={e} edge_attr_dim={d} unique_rows={uniq} (1 => repeated per relation)")

        # Compare cosine similarity with PID vs its English label
        pid = et[1]
        label = rel_labels.get(pid, pid)
        # Load a CLIP model whose text embedding dim matches edge_attr dim, if possible
        try:
            model, used_name, text_dim = _ensure_model_with_dim(d, args.clip_model, device)
        except Exception as ex:
            print(f"  CLIP load failed: {ex}. Skipping cosine checks.")
            continue
        if text_dim != d:
            print(f"  Warning: CLIP({used_name}) text dim {text_dim} != edge_attr_dim {d}. Skipping cosine checks.")
            continue
        with torch.no_grad():
            v_edge = edge_attr[0]  # representative row
            v_pid = encode_texts_with_clip([pid], model=model, device=device)[0]
            v_label = encode_texts_with_clip([label], model=model, device=device)[0]
        print(f"  cos(edge, PID='{pid}')={cosine_similarity(v_edge, v_pid):.4f} "
              f"cos(edge, label='{label}')={cosine_similarity(v_edge, v_label):.4f}")


def check_homo(data: "Data", container: dict, args: argparse.Namespace):
    edge_attr = getattr(data, "edge_attr", None)
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None or edge_index.dim() != 2 or edge_index.size(1) == 0:
        print("Homogeneous graph: no edges")
        return
    if edge_attr is None or edge_attr.numel() == 0:
        print("Homogeneous graph: edge_attr missing")
        return
    if edge_attr.dim() != 2:
        print(f"Homogeneous graph: edge_attr has unexpected shape {tuple(edge_attr.shape)}")
        return
    e = int(edge_index.size(1))
    d = int(edge_attr.size(1))
    try:
        uniq = int(torch.unique(edge_attr, dim=0).size(0))
    except Exception:
        uniq = -1
    print(f"Homogeneous: num_edges={e} edge_attr_dim={d} unique_rows={uniq} (1 => repeated per relation)")

    # If we have per-edge PID list, try a quick PID/label similarity check using the first edge
    pids = container.get("edge_index_label_en") or container.get("edge_property_ids")
    rel_labels = container.get("relation_labels_en") or container.get("property_labels_en") or {}
    if isinstance(pids, (list, tuple)) and len(pids) > 0:
        first = pids[0]
        pid = first if isinstance(first, str) else str(first)
        label = rel_labels.get(pid, pid)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model, used_name, text_dim = _ensure_model_with_dim(d, args.clip_model, device)
        except Exception as ex:
            print(f"  CLIP load failed: {ex}. Skipping cosine checks.")
            return
        if text_dim != d:
            print(f"  Warning: CLIP({used_name}) text dim {text_dim} != edge_attr_dim {d}. Skipping cosine checks.")
            return
        with torch.no_grad():
            v_edge = edge_attr[0]
            v_pid = encode_texts_with_clip([pid], model=model, device=device)[0]
            v_label = encode_texts_with_clip([label], model=model, device=device)[0]
        print(f"  cos(edge, PID='{pid}')={cosine_similarity(v_edge, v_pid):.4f} "
              f"cos(edge, label='{label}')={cosine_similarity(v_edge, v_label):.4f}")
    else:
        print("  No per-edge PID list found to compare against.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to .pt (may contain top-level dict with 'pyg_data')")
    parser.add_argument("--edge-type", default=None, help="Hetero edge type to check: src,rel,dst")
    parser.add_argument("--clip-model", default="auto", help="CLIP model name or 'auto' to match edge_attr_dim")
    args = parser.parse_args()

    if Data is None:
        print("ERROR: torch_geometric is not available.", file=sys.stderr)
        sys.exit(1)
    obj = torch.load(args.path, map_location="cpu")
    container = obj if isinstance(obj, dict) else {}
    data = None
    if isinstance(container, dict) and "pyg_data" in container:
        data = container["pyg_data"]
    elif isinstance(obj, (Data, HeteroData)):
        data = obj
    elif isinstance(container, dict):
        try:
            data = Data(**container)
        except Exception:
            print("Could not construct PyG Data from top-level dict; provide 'pyg_data' in the file.", file=sys.stderr)
            sys.exit(2)
    else:
        print("Unrecognized container; expected dict or PyG Data/HeteroData.", file=sys.stderr)
        sys.exit(2)

    if HeteroData is not None and isinstance(data, HeteroData):
        check_hetero(data, container, args)
    elif isinstance(data, Data):
        check_homo(data, container, args)
    else:
        print("Unsupported graph object type.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()


