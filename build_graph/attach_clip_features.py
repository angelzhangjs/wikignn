import argparse
import os
import sys
import torch
import re

try:
    from torch_geometric.data import Data, HeteroData  # type: ignore
except Exception:
    Data = None
    HeteroData = None

try:
    from data_utils import _coerce_edge_index  # reuse robust coercion
except Exception:
    _coerce_edge_index = None
# Optional progress bars
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None
try:
    import clip  # OpenAI CLIP: pip install git+https://github.com/openai/CLIP.git
except Exception:
    clip = None
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None

def _as_2d_float(t):
    t = torch.as_tensor(t)
    if t.dim() == 1:
        t = t.view(-1, 1)
    return t.to(torch.float)


def _pick_single(items):
    return items[0] if len(items) == 1 else None


def _is_pid(x) -> bool:
    return isinstance(x, str) and re.match(r"^P\d+$", x) is not None


def _build_edge_texts(container: dict, pid_list_key: str, rel_labels_key: str):
    if pid_list_key not in container:
        raise ValueError(f"Missing per-edge property list at '{pid_list_key}'.")
    pids = container[pid_list_key]
    rel_labels = container.get(rel_labels_key, {})
    texts = []
    for pid_or_label in pids:
        if _is_pid(pid_or_label):
            label = rel_labels.get(pid_or_label, pid_or_label)
            texts.append(str(label))
        else:
            texts.append(str(pid_or_label))
    return texts


def _get_entity_text_getter(container: dict, entity_text_key: str, entity_labels_key: str):
    # Prefer rich text; fall back to labels
    store = container.get(entity_text_key) or container.get(entity_labels_key)
    if store is None:
        raise ValueError(f"Missing entity text store: '{entity_text_key}' or '{entity_labels_key}'.")
    if isinstance(store, dict):
        def getter(i: int) -> str:
            key = str(int(i))
            return str(store.get(key, key))
        return getter
    else:
        def getter(i: int) -> str:
            idx = int(i)
            return str(store[idx]) if 0 <= idx < len(store) else str(idx)
        return getter


def _build_triple_texts_from_top_level(container: dict, pid_list_key: str, rel_labels_key: str,
                                       entity_text_key: str, entity_labels_key: str):
    if _coerce_edge_index is None:
        raise RuntimeError("data_utils._coerce_edge_index not available to parse top-level edge_index.")
    if 'edge_index' not in container:
        raise ValueError("Top-level 'edge_index' is required to build triple texts.")
    ei = _coerce_edge_index(container['edge_index'])
    if ei is None or ei.dim() != 2 or ei.size(0) != 2:
        raise ValueError("Unable to coerce top-level edge_index to [2, E].")
    pids = container.get(pid_list_key)
    if pids is None:
        raise ValueError(f"Missing per-edge property list at '{pid_list_key}'.")
    if ei.size(1) != len(pids):
        raise ValueError(f"edge_index edges ({ei.size(1)}) != {pid_list_key} length ({len(pids)}).")
    rel_labels = container.get(rel_labels_key, {})
    get_text = _get_entity_text_getter(container, entity_text_key, entity_labels_key)
    texts = []
    for i in range(ei.size(1)):
        s = int(ei[0, i].item())
        o = int(ei[1, i].item())
        pid = pids[i]
        r_txt = rel_labels.get(pid, pid)
        s_txt = get_text(s)
        o_txt = get_text(o)
        texts.append(f"{s_txt} {r_txt} {o_txt}")
    return texts


def _compose_entity_texts_by_index(container: dict, n_nodes: int,
                                   labels_key: str, descriptions_key: str, aliases_key: str,
                                   fallback_text_key: str | None = None):
    """
    Compose per-entity texts by index by concatenating label, description, and aliases when available.
    Expects dicts keyed by stringified indices. Falls back to an existing text_by_index map if provided.
    """
    texts = [""] * n_nodes
    labels = container.get(labels_key) if isinstance(container, dict) else None
    descs = container.get(descriptions_key) if isinstance(container, dict) else None
    aliases = container.get(aliases_key) if isinstance(container, dict) else None
    base = container.get(fallback_text_key) if (fallback_text_key and isinstance(container, dict)) else None
    for i in range(n_nodes):
        parts = []
        key = str(i)
        # label
        if isinstance(labels, dict) and key in labels and labels[key]:
            parts.append(str(labels[key]))
        # description
        if isinstance(descs, dict) and key in descs and descs[key]:
            parts.append(str(descs[key]))
        # aliases (list or dict)
        if isinstance(aliases, dict) and key in aliases and aliases[key]:
            val = aliases[key]
            if isinstance(val, (list, tuple)):
                parts.append("Aliases: " + ", ".join(str(x) for x in val[:10]))
            else:
                parts.append("Aliases: " + str(val))
        # base fallback
        if not parts and isinstance(base, dict) and key in base and base[key]:
            parts.append(str(base[key]))
        texts[i] = " ".join(parts) if parts else key
    return texts


def _encode_texts_clip(
    texts,
    model_name: str = "RN50",
    device: str | None = None,
    batch_size: int = 512,
    normalize: bool = True,
    max_chars: int | None = 512,
    show_progress: bool = False,
    progress_desc: str = "CLIP encoding",
):
    """
    Encode a list of texts with CLIP text encoder.
    - Truncates to CLIP's 77 token context via truncate=True.
    - Optionally caps raw string length to max_chars to avoid excessive tokenization costs.
    """
    if clip is None:
        raise RuntimeError("The 'clip' package is not installed. Install via: pip install git+https://github.com/openai/CLIP.git")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = clip.load(model_name, device=device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        iterator = range(0, len(texts), batch_size)
        if show_progress and tqdm is not None:
            total_batches = (len(texts) + batch_size - 1) // batch_size
            iterator = tqdm(iterator, total=total_batches, desc=progress_desc)
        for i in iterator:
            batch = []
            for t in texts[i:i + batch_size]:
                s = str(t).replace("\n", " ").strip()
                if max_chars is not None and len(s) > max_chars:
                    s = s[:max_chars]
                batch.append(s)
            toks = clip.tokenize(batch, truncate=True).to(device)
            feats = model.encode_text(toks).float()
            if normalize:
                feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu())
    return torch.cat(embeddings, dim=0)

def _encode_texts_sbert(
    texts,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str | None = None,
    batch_size: int = 512,
    normalize: bool = True,
    max_chars: int | None = 2048,
    show_progress: bool = False,
    progress_desc: str = "SBERT encoding",
):
    if SentenceTransformer is None:
        raise RuntimeError("The 'sentence-transformers' package is not installed. Install via: pip install sentence-transformers")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=device)
    proc = []
    for s in texts:
        s = str(s).replace("\n", " ").strip()
        if max_chars is not None and len(s) > max_chars:
            s = s[:max_chars]
        proc.append(s)
    embs = model.encode(
        proc,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=bool(show_progress and tqdm is not None),
        normalize_embeddings=False,
    ).to("cpu").float()
    if normalize:
        embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-12)
    return embs

def _encode_texts(
    texts,
    backend: str = "clip",
    clip_model: str = "RN50",
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str | None = None,
    batch_size: int = 512,
    normalize: bool = True,
    max_chars: int | None = 512,
    show_progress: bool = False,
    progress_desc: str = "Text encoding",
):
    if backend == "clip":
        return _encode_texts_clip(
            texts,
            model_name=clip_model,
            device=device,
            batch_size=batch_size,
            normalize=normalize,
            max_chars=max_chars,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )
    elif backend == "sbert":
        return _encode_texts_sbert(
            texts,
            model_name=sbert_model,
            device=device,
            batch_size=batch_size,
            normalize=normalize,
            max_chars=max_chars if max_chars is not None else 2048,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )
    else:
        raise ValueError(f"Unknown text backend: {backend}")


def attach_to_data(data: "Data", node_emb: torch.Tensor | None, edge_emb: torch.Tensor | None):
    if node_emb is not None:
        n = int(data.num_nodes) if getattr(data, 'num_nodes', None) is not None else int(data.x.size(0)) if getattr(data, 'x', None) is not None else None
        if n is None:
            raise ValueError("Cannot infer num_nodes to attach node embeddings.")
        if node_emb.size(0) != n:
            raise ValueError(f"Node embedding count {node_emb.size(0)} != num_nodes {n}.")
        data.x = node_emb
    if edge_emb is not None:
        ei = getattr(data, 'edge_index', None)
        if ei is None or ei.dim() != 2:
            raise ValueError("edge_index missing or invalid; cannot attach edge embeddings.")
        e = int(ei.size(1))
        if edge_emb.size(0) != e:
            raise ValueError(f"Edge embedding count {edge_emb.size(0)} != num_edges {e}.")
        data.edge_attr = edge_emb
    return data

def attach_to_hetero(data: "HeteroData", node_emb: torch.Tensor | None, edge_emb: torch.Tensor | None,
                     node_type: str | None, edge_type: str | None):
    # Node embeddings
    if node_emb is not None:
        candidates = []
        for nt in data.node_types:
            store = data[nt]
            try:
                n = int(store.num_nodes)
            except Exception:
                x = getattr(store, 'x', None)
                n = int(x.size(0)) if x is not None else None
            if n is not None and n == node_emb.size(0):
                candidates.append(nt)
        chosen = node_type if node_type is not None else _pick_single(candidates)
        if chosen is None:
            raise ValueError(f"Ambiguous node type for embeddings; candidates={candidates}. Specify --node-type.")
        data[chosen].x = node_emb
    # Edge embeddings
    if edge_emb is not None:
        candidates = []
        for et in data.edge_types:
            ei = getattr(data[et], 'edge_index', None)
            if ei is not None and ei.dim() == 2 and int(ei.size(1)) == edge_emb.size(0):
                candidates.append(et)
        if edge_type is not None:
            # edge_type arg format: src,rel,dst
            parts = tuple(edge_type.split(','))
            if len(parts) != 3:
                raise ValueError("Invalid --edge-type format. Use src,rel,dst")
            chosen = parts
        else:
            chosen = _pick_single(candidates)
        if chosen is None:
            raise ValueError(f"Ambiguous edge type for embeddings; candidates={candidates}. Specify --edge-type as src,rel,dst.")
        data[chosen].edge_attr = edge_emb
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to input .pt")
    parser.add_argument("--out", required=False, help="Path to output .pt (default: <path>.with_clip.pt)")
    parser.add_argument("--node-emb-key", default="clip_text_node_emb_en", help="Top-level key for node embeddings")
    parser.add_argument("--edge-emb-key", default="clip_text_edge_mean_emb_en", help="Top-level key for edge embeddings")
    parser.add_argument("--node-type", default=None, help="HeteroData node type to attach to")
    parser.add_argument("--edge-type", default=None, help="HeteroData edge type to attach to (format: src,rel,dst)")
    # New: compute edge embeddings from per-edge property IDs using CLIP text encoder
    parser.add_argument("--compute-edge-clip", action="store_true", help="Compute edge embeddings from per-edge property IDs via CLIP text encoder")
    parser.add_argument("--pid-list-key", default="edge_index_label_en", help="Top-level key listing per-edge property IDs/labels")
    parser.add_argument("--rel-labels-key", default="relation_labels_en", help="Top-level key mapping PID->English label")
    parser.add_argument("--edge-text-mode", choices=["pid", "label", "triple"], default="pid", help="How to build per-edge texts for CLIP")
    parser.add_argument("--entity-text-key", default="entity_text_en_by_index", help="Top-level key for per-entity rich text")
    parser.add_argument("--entity-labels-key", default="entity_labels_en", help="Top-level key for per-entity labels (fallback)")
    parser.add_argument("--entity-descriptions-key", default="entity_descriptions_en", help="Top-level key for per-entity descriptions (optional)")
    parser.add_argument("--entity-aliases-key", default="entity_aliases_en", help="Top-level key for per-entity aliases (optional)")
    parser.add_argument("--entity-meta", default="", help="Optional path to entity metadata .pt to source texts by index")
    parser.add_argument("--compute-node-clip", action="store_true", help="Compute node embeddings from per-entity text via CLIP")
    parser.add_argument("--text-backend", choices=["clip", "sbert"], default="clip", help="Text embedding backend to use")
    parser.add_argument("--clip-model", default="RN50", help="CLIP model name for text encoder (e.g., RN50, ViT-B/32)")
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence-Transformers model name")
    parser.add_argument("--clip-batch-size", type=int, default=512, help="Batch size for CLIP text encoding")
    parser.add_argument("--clip-max-chars", type=int, default=512, help="Cap input text length (chars) before tokenization")
    parser.add_argument("--no-normalize", action="store_true", help="Do not L2-normalize CLIP embeddings")
    parser.add_argument("--progress", action="store_true", help="Show tqdm progress bars during CLIP encoding")
    # New: attach per-relation embeddings to every hetero edge type
    parser.add_argument("--attach-all-hetero-rel", action="store_true", help="Attach per-relation CLIP embeddings to all hetero edge types")
    parser.add_argument("--rel-emb-key", default="clip_text_rel_emb_en_by_pid", help="Top-level key for per-relation CLIP embeddings dict")
    parser.add_argument("--compute-rel-clip", action="store_true", help="Compute per-relation CLIP embeddings from relation labels")
    args = parser.parse_args()

    if Data is None:
        print("ERROR: torch_geometric not available.", file=sys.stderr)
        sys.exit(1)

    obj = torch.load(args.path, map_location="cpu")
    contains = obj if isinstance(obj, dict) else {}
    meta = None
    if args.entity_meta:
        try:
            meta = torch.load(args.entity_meta, map_location="cpu")
        except Exception:
            meta = None

    node_emb = _as_2d_float(contains[args.node_emb_key]) if isinstance(contains, dict) and args.node_emb_key in contains else None
    edge_emb = None
    if isinstance(contains, dict) and args.edge_emb_key in contains and torch.is_tensor(contains[args.edge_emb_key]):
        edge_emb = _as_2d_float(contains[args.edge_emb_key])
    # Optionally compute edge embeddings from property IDs using CLIP text encoder
    if args.compute_edge_clip:
        # In hetero triple mode with --attach-all-hetero-rel, we compute per-edge embeddings later per edge type.
        defer_hetero_triple = args.attach_all_hetero_rel and args.edge_text_mode == "triple"
        if not defer_hetero_triple:
            if args.edge_text_mode == "triple":
                texts = _build_triple_texts_from_top_level(
                    contains, args.pid_list_key, args.rel_labels_key,
                    args.entity_text_key, args.entity_labels_key
                )
            elif args.edge_text_mode == "label":
                # Map PID->label per-edge
                texts = _build_edge_texts(contains, args.pid_list_key, args.rel_labels_key)
            else:
                # PID-only strings
                pids = contains.get(args.pid_list_key)
                if pids is None:
                    raise ValueError(f"Missing per-edge property list at '{args.pid_list_key}'.")
                texts = [str(x) for x in pids]
            edge_emb = _encode_texts(
                texts,
                backend=args.text_backend,
                clip_model=args.clip_model,
                sbert_model=args.sbert_model,
                device=None,
                batch_size=args.clip_batch_size,
                normalize=not args.no_normalize,
                max_chars=args.clip_max_chars,
                show_progress=args.progress,
                progress_desc="Text edge embeddings",
            )
            # Save computed tensor at top level for future reuse
            if isinstance(contains, dict):
                obj[args.edge_emb_key] = edge_emb
    # Optionally compute node embeddings from per-entity text via CLIP
    if args.compute_node_clip:
        # Determine number of nodes for selected node type (default: entity)
        # We'll compute after we construct 'data' to know num_nodes
        pass

    # Determine graph object
    data = None
    if isinstance(contains, dict) and 'pyg_data' in contains:
        data = contains['pyg_data']
    elif isinstance(obj, (Data, HeteroData)):
        data = obj
    elif isinstance(contains, dict):
        try:
            data = Data(**contains)
        except Exception:
            print("Could not construct PyG Data from top-level dict; provide 'pyg_data' in the file.", file=sys.stderr)
            sys.exit(2)
    else:
        print("Unrecognized container; expected dict or PyG Data/HeteroData.", file=sys.stderr)
        sys.exit(2)

    # Proceed if embeddings are present OR we are asked to compute them now/later
    if node_emb is None and edge_emb is None:
        will_compute_any = bool(args.compute_node_clip or args.compute_edge_clip or args.compute_rel_clip)
        if not will_compute_any:
            print("No embeddings found at top-level with the given keys, and no compute flags provided.", file=sys.stderr)
            sys.exit(3)

    if HeteroData is not None and isinstance(data, HeteroData):
        # If computing node embeddings, build texts by index for the chosen node type now
        if args.compute_node_clip:
            target_nt = args.node_type or "entity"
            if target_nt not in data.node_types:
                raise ValueError(f"Node type '{target_nt}' not present in graph.")
            try:
                n_nodes = int(data[target_nt].num_nodes)
            except Exception:
                raise ValueError(f"Cannot infer num_nodes for node type '{target_nt}'.")
            # Try to source texts by index from graph payload first, then meta
            texts_by_index = None
            for source in (contains, meta):
                if isinstance(source, dict) and args.entity_text_key in source:
                    mapping = source[args.entity_text_key]
                    if isinstance(mapping, dict):
                        tmp = [""] * n_nodes
                        for k, v in mapping.items():
                            try:
                                idx = int(k)
                                if 0 <= idx < n_nodes:
                                    tmp[idx] = str(v)
                            except Exception:
                                continue
                        texts_by_index = tmp
                        break
            if texts_by_index is None and isinstance(contains, dict):
                # Fallback 1: invert node_index mapping for this node type (e.g., 'literal', 'external_id')
                node_index = contains.get("node_index")
                if isinstance(node_index, dict):
                    source_map = node_index.get(target_nt)
                    if isinstance(source_map, dict):
                        tmp = [""] * n_nodes
                        for text_value, idx in source_map.items():
                            try:
                                idx_i = int(idx)
                            except Exception:
                                continue
                            if 0 <= idx_i < n_nodes:
                                tmp[idx_i] = str(text_value)
                        if any(s != "" for s in tmp):
                            texts_by_index = tmp
                # Fallback 2: look for a '{node_type}_text_*_by_index' mapping
                if texts_by_index is None:
                    for cand_key in (f"{target_nt}_text_en_by_index", f"{target_nt}_text_by_index"):
                        cand = contains.get(cand_key)
                        if isinstance(cand, dict):
                            tmp = [""] * n_nodes
                            for k, v in cand.items():
                                try:
                                    idx = int(k)
                                    if 0 <= idx < n_nodes:
                                        tmp[idx] = str(v)
                                except Exception:
                                    continue
                            texts_by_index = tmp
                            break
            if texts_by_index is None:
                raise ValueError(
                    f"Missing texts for node type '{target_nt}'. Provide '{args.entity_text_key}', "
                    f"or ensure 'node_index.{target_nt}' exists to derive texts, "
                    f"or add a '{target_nt}_text_en_by_index' map."
                )
            node_emb = _encode_texts(
                texts_by_index,
                backend=args.text_backend,
                clip_model=args.clip_model,
                sbert_model=args.sbert_model,
                device=None,
                batch_size=args.clip_batch_size,
                normalize=not args.no_normalize,
                max_chars=args.clip_max_chars,
                show_progress=args.progress,
                progress_desc=f"Text node embeddings [{target_nt}]",
            )
            if isinstance(contains, dict):
                obj[args.node_emb_key] = node_emb
        # Option A: attach one edge_emb to a specific hetero edge type
        data = attach_to_hetero(data, node_emb, edge_emb, args.node_type, args.edge_type)
        # Option B: attach embeddings to every hetero edge store
        if args.attach_all_hetero_rel:
            # Build lookup of relation -> embedding vector
            rel_lookup = {}
            rel_emb_container = contains.get(args.rel_emb_key) if isinstance(contains, dict) else None
            if isinstance(rel_emb_container, dict) and len(rel_emb_container) > 0:
                for pid, vec in rel_emb_container.items():
                    try:
                        rel_lookup[pid] = torch.as_tensor(vec, dtype=torch.float).view(-1)
                    except Exception:
                        continue
            # Optionally compute for missing relations using CLIP text encoder
            rel_labels = contains.get(args.rel_labels_key) if isinstance(contains, dict) else None
            def _ensure_rel_vec(pid: str):
                if pid in rel_lookup:
                    return rel_lookup[pid]
                label_text = None
                if isinstance(rel_labels, dict) and pid in rel_labels:
                    label_text = str(rel_labels[pid])
                elif args.compute_rel_clip:
                    label_text = pid
                if args.compute_rel_clip and label_text is not None:
                    vec = _encode_texts(
                        [label_text],
                        backend=args.text_backend,
                        clip_model=args.clip_model,
                        sbert_model=args.sbert_model,
                        normalize=not args.no_normalize,
                        max_chars=args.clip_max_chars,
                        show_progress=False,
                        progress_desc="Text relation embedding",
                    )
                    rel_lookup[pid] = vec[0].view(-1)
                    return rel_lookup[pid]
                return None
            # Attach to all hetero edge stores
            for et in data.edge_types:
                store = data[et]
                ei = getattr(store, 'edge_index', None)
                if ei is None or ei.dim() != 2 or ei.size(1) == 0:
                    continue
                pid = et[1]
                # Per-edge triple mode: build unique text per edge and encode
                if args.edge_text_mode == "triple":
                    rel_labels_map = contains.get(args.rel_labels_key, {})
                    r_txt = rel_labels_map.get(pid, pid)
                    # Compose per-entity texts by index (label + desc + aliases), falling back to text_by_index
                    try:
                        n_all = 0
                        # Infer total nodes for 'entity' type if available
                        if "entity" in data.node_types:
                            n_all = int(data["entity"].num_nodes)
                        texts_by_index = _compose_entity_texts_by_index(
                            contains,
                            n_all if n_all > 0 else int(ei.max().item() + 1),
                            args.entity_labels_key,
                            args.entity_descriptions_key,
                            args.entity_aliases_key,
                            fallback_text_key=args.entity_text_key,
                        )
                    except Exception:
                        # Fallback to simple getter if composition fails
                        get_text = _get_entity_text_getter(contains, args.entity_text_key, args.entity_labels_key)
                        texts_by_index = None
                    texts = []
                    for i in range(int(ei.size(1))):
                        s = int(ei[0, i].item())
                        o = int(ei[1, i].item())
                        if texts_by_index is not None and s < len(texts_by_index) and o < len(texts_by_index):
                            s_txt = texts_by_index[s]
                            o_txt = texts_by_index[o]
                        else:
                            s_txt = get_text(s) if 'get_text' in locals() else str(s)
                            o_txt = get_text(o) if 'get_text' in locals() else str(o)
                        texts.append(f"{s_txt} {r_txt} {o_txt}")
                    vecs = _encode_texts(
                        texts,
                        backend=args.text_backend,
                        clip_model=args.clip_model,
                        sbert_model=args.sbert_model,
                        device=None,
                        batch_size=args.clip_batch_size,
                        normalize=not args.no_normalize,
                        max_chars=args.clip_max_chars,
                        show_progress=args.progress,
                        progress_desc=f"Text edges [{et[1]}]",
                    )
                    store.edge_attr = vecs
                else:
                    # Per-relation vector repeated
                    vec = _ensure_rel_vec(pid)
                    if vec is None:
                        continue
                    store.edge_attr = vec.view(1, -1).repeat(int(ei.size(1)), 1)
    elif isinstance(data, Data):
        data = attach_to_data(data, node_emb, edge_emb)
    else:
        print("Unsupported graph object type.", file=sys.stderr)
        sys.exit(2)

    out_path = args.out or (args.path + ".with_clip.pt")
    if isinstance(obj, dict) and 'pyg_data' in obj:
        obj['pyg_data'] = data
        torch.save(obj, out_path)
    else:
        torch.save(data, out_path)
    print(f"Saved graph with attached embeddings to {out_path}")


if __name__ == "__main__":
    main()


