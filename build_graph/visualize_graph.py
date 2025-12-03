import os
import math
import torch
import matplotlib.pyplot as plt
import networkx as nx
from data_utils import load_graph
from torch_geometric.utils import to_networkx, degree, subgraph


def _pick_subgraph_nodes(edge_index, num_nodes: int, max_nodes: int):
    if num_nodes <= max_nodes:
        return torch.arange(num_nodes)
    # Fall back to simple head-of-range if edges are unavailable
    if edge_index is None or not (torch.is_tensor(edge_index) and edge_index.dim() == 2 and edge_index.size(0) == 2):
        return torch.arange(min(max_nodes, num_nodes))
    deg = degree(edge_index[0], num_nodes=num_nodes)
    topk = min(max_nodes, num_nodes)
    _, idx = torch.topk(deg, k=topk, largest=True, sorted=False)
    return idx.sort().values


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _build_hetero_networkx(data):
    G = nx.Graph()
    # Add nodes per type with composite IDs
    for node_type in getattr(data, 'node_types', []):
        store = data[node_type]
        try:
            n = store.num_nodes
        except Exception:
            n = getattr(store, 'x', None).size(0) if getattr(store, 'x', None) is not None else 0
        for i in range(int(n)):
            G.add_node((node_type, i), node_type=node_type)
    # Add edges per relation
    for (src, rel, dst) in getattr(data, 'edge_types', []):
        store = data[(src, rel, dst)]
        ei = getattr(store, 'edge_index', None)
        if ei is None or ei.numel() == 0:
            continue
        rows, cols = ei[0].tolist(), ei[1].tolist()
        for u, v in zip(rows, cols):
            G.add_edge((src, int(u)), (dst, int(v)), rel=rel, edge_type=(src, rel, dst))
    return G

def _pick_hetero_nodes_by_degree(G: "nx.Graph", max_nodes: int):
    if G.number_of_nodes() <= max_nodes:
        return list(G.nodes())
    deg_dict = dict(G.degree())
    # Sort nodes by degree descending
    nodes_sorted = sorted(deg_dict.keys(), key=lambda n: deg_dict[n], reverse=True)
    return nodes_sorted[:max_nodes]

if __name__ == '__main__':
    _ensure_dir('graph_visualization')
    data = load_graph('graph_output/clean_graph.pyg_en_labeled.pt')
    max_nodes = 1200

    # Heterogeneous branch
    if hasattr(data, 'node_types') and hasattr(data, 'edge_types'):
        G = _build_hetero_networkx(data)
        nodes_keep = _pick_hetero_nodes_by_degree(G, max_nodes)
        G = G.subgraph(nodes_keep).copy()
        pos = nx.spring_layout(G, seed=42, k=1.0 / math.sqrt(max(1, G.number_of_nodes())))
        # Color by node type
        node_types = sorted({G.nodes[n].get('node_type', 'node') for n in G.nodes()})
        type_to_idx = {t: i for i, t in enumerate(node_types)}
        cmap = plt.cm.get_cmap('tab20', max(1, len(node_types)))
        node_color = [cmap(type_to_idx[G.nodes[n].get('node_type', 'node')]) for n in G.nodes()]
        plt.figure(figsize=(10, 10), dpi=160)
        nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=0.5, alpha=0.6)
        nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=12, linewidths=0.0)
        plt.axis('off')
        title = f"Hetero graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(node_types)} node types)"
        plt.title(title)
        out_path = os.path.join('graph_visualization', 'graph_viz.png')
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"Saved visualization to {out_path}")
    else:
        # Homogeneous branch
        num_nodes = data.x.size(0)
        nodes = _pick_subgraph_nodes(data.edge_index, num_nodes, max_nodes)
        if data.edge_index is not None and torch.is_tensor(data.edge_index) and data.edge_index.dim() == 2 and data.edge_index.size(0) == 2:
            ei, _ = subgraph(nodes, data.edge_index, relabel_nodes=True)
        else:
            ei = torch.empty((2, 0), dtype=torch.long)
        x_sub = data.x[nodes]
        y_sub = data.y[nodes] if getattr(data, 'y', None) is not None else None
        data_sub = data.__class__(x=x_sub, edge_index=ei, y=y_sub)
        G = to_networkx(data_sub, to_undirected=True)
        pos = nx.spring_layout(G, seed=42, k=1.0 / math.sqrt(max(1, G.number_of_nodes())))
        if y_sub is not None:
            num_classes = int(y_sub.max().item() + 1)
            colors = [int(y_sub[n].item()) for n in range(y_sub.size(0))]
            cmap = plt.cm.get_cmap('tab20', num_classes)
            node_color = [cmap(c) for c in colors]
        else:
            degs = [G.degree(n) for n in G.nodes()]
            vmax = max(degs) if len(degs) > 0 else 1
            node_color = [plt.cm.Blues(d / max(vmax, 1)) for d in degs]
        plt.figure(figsize=(10, 10), dpi=160)
        nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=0.5, alpha=0.7)
        nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=12, linewidths=0.0)
        plt.axis('off')
        title = f"Graph visualization ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)"
        plt.title(title)
        out_path = os.path.join('graph_visualization', 'graph_viz.png')
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"Saved visualization to {out_path}")

