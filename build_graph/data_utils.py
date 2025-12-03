import torch
from torch_geometric.data import Data
try:
    from torch_geometric.data import HeteroData  # type: ignore
except Exception:  # pragma: no cover
    HeteroData = None  # fallback if not available
from torch_geometric.utils import degree


def _coerce_edge_index(edge_repr):
    if edge_repr is None:
        return None
    if torch.is_tensor(edge_repr):
        if edge_repr.dim() == 2:
            if edge_repr.size(0) == 2:
                return edge_repr.to(torch.long)
            if edge_repr.size(1) == 2:
                return edge_repr.t().contiguous().to(torch.long)
        return None
    if isinstance(edge_repr, dict):
        if 'edge_index' in edge_repr:
            return _coerce_edge_index(edge_repr['edge_index'])
        # Numeric keyed dicts: {0: rows, 1: cols} or {'0': rows, '1': cols}
        if 0 in edge_repr and 1 in edge_repr:
            row = torch.as_tensor(edge_repr[0], dtype=torch.long)
            col = torch.as_tensor(edge_repr[1], dtype=torch.long)
            return torch.stack([row, col], dim=0)
        if '0' in edge_repr and '1' in edge_repr:
            row = torch.as_tensor(edge_repr['0'], dtype=torch.long)
            col = torch.as_tensor(edge_repr['1'], dtype=torch.long)
            return torch.stack([row, col], dim=0)
        key_pairs = [
            ('row', 'col'),
            ('rows', 'cols'),
            ('src', 'dst'),
            ('source', 'target'),
            ('u', 'v'),
        ]
        for k1, k2 in key_pairs:
            if k1 in edge_repr and k2 in edge_repr:
                row = torch.as_tensor(edge_repr[k1], dtype=torch.long)
                col = torch.as_tensor(edge_repr[k2], dtype=torch.long)
                return torch.stack([row, col], dim=0)
        for key in ['edges', 'edge_list', 'indices']:
            if key in edge_repr:
                return _coerce_edge_index(torch.as_tensor(edge_repr[key]))
        # Nx2 arrays under common keys
        for key in ['values', 'data', 'pairs']:
            if key in edge_repr:
                tensor = torch.as_tensor(edge_repr[key])
                return _coerce_edge_index(tensor)
        return None
    if isinstance(edge_repr, (list, tuple)):
        if len(edge_repr) == 2 and all(isinstance(x, (list, tuple, torch.Tensor)) for x in edge_repr):
            row = torch.as_tensor(edge_repr[0], dtype=torch.long)
            col = torch.as_tensor(edge_repr[1], dtype=torch.long)
            return torch.stack([row, col], dim=0)
        try:
            as_tensor = torch.as_tensor(edge_repr)
            return _coerce_edge_index(as_tensor)
        except Exception:
            return None
    return None


def _infer_num_nodes(data, loaded, edge_index):
    if getattr(data, 'num_nodes', None) is not None:
        try:
            return int(data.num_nodes)
        except Exception:
            pass
    if getattr(data, 'x', None) is not None:
        return int(data.x.size(0))
    if getattr(data, 'y', None) is not None:
        return int(data.y.size(0))
    if edge_index is not None and torch.is_tensor(edge_index) and edge_index.numel() > 0:
        return int(edge_index.max().item() + 1)
    if isinstance(loaded, dict):
        for k in ['num_nodes', 'n_nodes', 'number_of_nodes']:
            if k in loaded and loaded[k] is not None:
                try:
                    return int(loaded[k])
                except Exception:
                    pass
    return None


def _infer_num_nodes_hetero_node(data, node_type: str):
    # Try store-provided num_nodes
    try:
        n = getattr(data[node_type], 'num_nodes', None)
        if n is not None:
            return int(n)
    except Exception:
        pass
    # Infer from incident edges
    max_idx = -1
    for (src, _, dst) in getattr(data, 'edge_types', []):
        store = data[(src, _, dst)]
        ei = getattr(store, 'edge_index', None)
        if ei is None:
            continue
        if src == node_type and ei.numel() > 0:
            max_idx = max(max_idx, int(ei[0].max().item()))
        if dst == node_type and ei.numel() > 0:
            max_idx = max(max_idx, int(ei[1].max().item()))
    return (max_idx + 1) if max_idx >= 0 else None


def _coerce_hetero_edges_inplace(data: "HeteroData"):
    # Ensure each relation store has a proper [2, E] LongTensor edge_index
    for edge_type in getattr(data, 'edge_types', []):
        store = data[edge_type]
        raw = getattr(store, 'edge_index', None)
        coerced = _coerce_edge_index(raw)
        if coerced is not None and torch.is_tensor(coerced) and coerced.dim() == 2 and coerced.size(0) == 2:
            store.edge_index = coerced


def _ensure_hetero_features_inplace(data: "HeteroData", loaded: dict):
    # Try to map top-level feature arrays to matching node types by length
    candidate_feature_keys = [
        'x', 'feat', 'features', 'node_feat', 'node_features', 'node_attr',
        'clip_text_node_emb_en', 'node_emb', 'node_embeddings'
    ]
    top_level_feats = {}
    if isinstance(loaded, dict):
        for key in candidate_feature_keys:
            if key in loaded and loaded[key] is not None:
                try:
                    tensor = torch.as_tensor(loaded[key], dtype=torch.float)
                    if tensor.dim() == 1:
                        tensor = tensor.view(-1, 1)
                    top_level_feats[key] = tensor
                except Exception:
                    continue

    # Assign features per node type
    for node_type in getattr(data, 'node_types', []):
        store = data[node_type]
        x = getattr(store, 'x', None)
        if x is not None:
            # Ensure 2D
            if x.dim() == 1:
                store.x = x.view(-1, 1)
            continue
        n = _infer_num_nodes_hetero_node(data, node_type)
        assigned = False
        if n is not None:
            for key, tensor in top_level_feats.items():
                if tensor.size(0) == n:
                    store.x = tensor
                    assigned = True
                    break
        if assigned:
            continue
        # Fallback: degree-based features aggregated over in/out edges
        if n is None:
            # Cannot infer; skip feature assignment for this node type
            continue
        deg = torch.zeros(n, dtype=torch.float)
        for (src, rel, dst) in getattr(data, 'edge_types', []):
            ei = getattr(data[(src, rel, dst)], 'edge_index', None)
            if ei is None or ei.numel() == 0:
                continue
            if src == node_type:
                one = torch.ones(ei.size(1), dtype=torch.float)
                deg.index_add_(0, ei[0].to(torch.long), one)
            if dst == node_type:
                one = torch.ones(ei.size(1), dtype=torch.float)
                deg.index_add_(0, ei[1].to(torch.long), one)
        store.x = deg.view(-1, 1)

    # Normalize labels per node type when present
    for node_type in getattr(data, 'node_types', []):
        store = data[node_type]
        y = getattr(store, 'y', None)
        if y is None and isinstance(loaded, dict):
            # Try top-level labels that match length
            for key in ['y', 'label', 'labels', 'targets', 'node_labels']:
                if key in loaded and loaded[key] is not None:
                    cand = torch.as_tensor(loaded[key])
                    n = _infer_num_nodes_hetero_node(data, node_type)
                    if n is not None and cand.size(0) == n:
                        y = cand
                        break
        if y is not None:
            if y.dim() > 1:
                y = y.view(-1)
            if y.dtype != torch.long:
                y = y.long()
            store.y = y


def load_graph(path: str):
    loaded = torch.load(path, map_location='cpu')
    # If a nested pyg_data exists, prefer it as the base
    if isinstance(loaded, dict) and 'pyg_data' in loaded:
        inner = loaded['pyg_data']
        if isinstance(inner, dict):
            data = Data(**inner)
        else:
            data = inner
        # Keep outer dict around for fallbacks/metadata
    else:
        data = Data(**loaded) if isinstance(loaded, dict) else loaded

    # Heterogeneous graphs: keep hetero, normalize per-type features/edges
    if HeteroData is not None and isinstance(data, HeteroData):
        _coerce_hetero_edges_inplace(data)
        if isinstance(loaded, dict):
            _ensure_hetero_features_inplace(data, loaded)
        return data

    # Homogeneous graphs
    coerced_ei = None
    if getattr(data, 'edge_index', None) is not None:
        coerced_ei = _coerce_edge_index(data.edge_index)
    if coerced_ei is None and isinstance(loaded, dict):
        for key in ['edge_index', 'edges', 'edge_list', 'indices']:
            if key in loaded and loaded[key] is not None:
                coerced_ei = _coerce_edge_index(loaded[key])
                if coerced_ei is not None:
                    break
        if coerced_ei is None:
            coerced_ei = _coerce_edge_index(loaded)
    # Try nested pyg_data for edge_index if still missing
    if coerced_ei is None and isinstance(loaded, dict) and 'pyg_data' in loaded:
        coerced_ei = _coerce_edge_index(loaded['pyg_data'])
    if coerced_ei is not None and torch.is_tensor(coerced_ei) and coerced_ei.dim() == 2 and coerced_ei.size(0) == 2:
        data.edge_index = coerced_ei
    else:
        if hasattr(data, 'edge_index'):
            data.edge_index = None

    if getattr(data, 'x', None) is None and isinstance(loaded, dict):
        for key in ['x', 'feat', 'features', 'node_feat', 'node_features', 'node_attr',
                    'clip_text_node_emb_en', 'node_emb', 'node_embeddings']:
            if key in loaded and loaded[key] is not None:
                data.x = torch.as_tensor(loaded[key], dtype=torch.float)
                break
    if getattr(data, 'x', None) is None and isinstance(loaded, dict) and 'pyg_data' in loaded:
        inner = loaded['pyg_data']
        try:
            cand = inner.get('x') if isinstance(inner, dict) else getattr(inner, 'x', None)
        except Exception:
            cand = None
        if cand is not None:
            data.x = torch.as_tensor(cand, dtype=torch.float)

    if getattr(data, 'y', None) is None and isinstance(loaded, dict):
        for key in ['y', 'label', 'labels', 'targets', 'node_labels']:
            if key in loaded and loaded[key] is not None:
                data.y = torch.as_tensor(loaded[key])
                break
    if getattr(data, 'y', None) is None and isinstance(loaded, dict) and 'pyg_data' in loaded:
        inner = loaded['pyg_data']
        try:
            cand = inner.get('y') if isinstance(inner, dict) else getattr(inner, 'y', None)
        except Exception:
            cand = None
        if cand is not None:
            data.y = torch.as_tensor(cand)

    if getattr(data, 'x', None) is None and torch.is_tensor(getattr(data, 'edge_index', None)) and data.edge_index.dim() == 2 and data.edge_index.size(0) == 2:
        num_nodes = _infer_num_nodes(data, loaded, data.edge_index)
        if num_nodes is None:
            raise ValueError("Cannot infer number of nodes to build degree features.")
        deg = degree(data.edge_index[0].to(torch.long), num_nodes=num_nodes)
        data.x = deg.view(-1, 1).to(torch.float)

    if getattr(data, 'x', None) is not None and data.x.dim() == 1:
        data.x = data.x.view(-1, 1)
    if getattr(data, 'y', None) is not None:
        if data.y.dim() > 1:
            data.y = data.y.view(-1)
        if data.y.dtype != torch.long:
            data.y = data.y.long()

    if getattr(data, 'x', None) is None:
        available_keys = list(loaded.keys()) if isinstance(loaded, dict) else []
        raise ValueError(
            "Node features not found. Provide one of ['x','feat','features','node_feat','node_features'] "
            "in the saved object. Alternatively, include a usable 'edge_index' convertible to a "
            "[2, E] tensor so degree features can be constructed. "
            f"Available keys: {available_keys}"
        )
    return data


