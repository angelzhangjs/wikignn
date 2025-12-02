import os
import matplotlib.pyplot as plt
import networkx as nx
from itertools import cycle
from collections import Counter, deque
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_networkx
import torch_geometric.transforms as T
import random

# Optionally build a NetworkX graph directly from a payload dict
def _build_networkx_from_payload(payload):
    """
    Supports payloads shaped like:
      {
        "node_index": {"entity": {...}, "literal": {...}, ...} (optional),
        "edge_index": {"srcType:prop:dstType": Tensor[[...],[...]], ...},
        "edge_claim_ids": {"srcType:prop:dstType": [str|None, ...]} (optional)
      }
    """
    if not isinstance(payload, dict):
        return None
    edge_index = payload.get("edge_index")
    if not isinstance(edge_index, dict):
        return None
    G_local = nx.MultiDiGraph()

    def _coerce_int(value):
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    def _as_two_lists(ei):
        if ei is None:
            return None, None
        try:
            if hasattr(ei, "tolist"):
                src = ei[0].tolist()
                dst = ei[1].tolist()
            else:
                src = list(ei[0])
                dst = list(ei[1])
            return src, dst
        except Exception:
            return None, None
    for key_str, ei in edge_index.items():
        try:
            src_t, prop, dst_t = key_str.split(":")
        except Exception:
            # Fallback if key doesn't follow expected pattern
            src_t, prop, dst_t = "unknown", str(key_str), "unknown"
        # Tensors or lists supported; tolerate None/nulls
        s_list, d_list = _as_two_lists(ei)
        if s_list is None or d_list is None:
            continue
        for s_idx, d_idx in zip(s_list, d_list):
            si = _coerce_int(s_idx)
            di = _coerce_int(d_idx)
            if si is None or di is None:
                continue
            u = (src_t, si)
            v = (dst_t, di)
            if u not in G_local:
                G_local.add_node(u, node_type=src_t if src_t else "unknown")
            if v not in G_local:
                G_local.add_node(v, node_type=dst_t if dst_t else "unknown")
            # Edge attributes; ensure strings for coloring/stats
            etype = key_str if key_str is not None else "unknown"
            p = prop if prop is not None else "unknown"
            G_local.add_edge(u, v, edge_type=str(etype), property_id=str(p))
    return G_local

# Choose attribute to define node type
def get_type(n):
    return G.nodes[n].get('node_type') or G.nodes[n].get('node_kind') or 'unknown'

# Load PyG graph (.pyg.pt)
PYG_GRAPH_PATH = os.environ.get("PYG_GRAPH_PATH", "/home/ghr/angel/gnn/graph_output/graph.pyg.pt")
GRAPH_VIZ_DIR = os.environ.get("GRAPH_VIZ_DIR", "/home/ghr/angel/gnn/graph_visualization")
obj = torch.load(PYG_GRAPH_PATH, map_location="cpu")

# Try to coerce the loaded object into a PyG Data/HeteroData or directly to NetworkX
data = None
G = None
if isinstance(obj, (Data, HeteroData)):
    data = obj
elif isinstance(obj, dict):
    if "data" in obj and isinstance(obj["data"], (Data, HeteroData)):
        data = obj["data"]
    elif "data" in obj and isinstance(obj["data"], dict):
        # Reconstruct Data from a plain dict
        try:
            data = Data.from_dict(obj["data"])  # type: ignore[attr-defined]
        except Exception:
            try:
                data = Data(**obj["data"])
            except Exception:
                pass
    elif "edge_index" in obj and isinstance(obj["edge_index"], dict):
        # Looks like a payload produced by a custom exporter; build NX directly
        G = _build_networkx_from_payload(obj)
    elif "edge_index" in obj:
        # Plain homogeneous dict representation
        try:
            data = Data.from_dict(obj)  # type: ignore[attr-defined]
        except Exception:
            try:
                data = Data(**obj)
            except Exception:
                pass

# Build NetworkX graph
if data is not None:
    # Normalize to homogeneous undirected Data
    if isinstance(data, HeteroData):
        data = T.ToHomogeneous()(data)
    if isinstance(data, Data):
        data = T.ToUndirected()(data)
    else:
        raise TypeError(f"Unsupported graph object type: {type(data)}")

    # Prepare attrs to carry over
    node_attrs = []
    for attr in ("y", "node_type", "node_kind"):
        if hasattr(data, attr):
            node_attrs.append(attr)
    edge_attrs = []
    for attr in ("edge_type", "edge_weight", "edge_attr", "edge_kind", "property_id"):
        if hasattr(data, attr):
            edge_attrs.append(attr)

    # Convert to NetworkX
    G = to_networkx(
        data,
        to_undirected=True,
        node_attrs=node_attrs or None,
        edge_attrs=edge_attrs or None
    )

if G is None:
    raise TypeError(f"Unsupported graph object type: {type(obj)}")

# Sample a smaller subgraph (BFS) for visualization if requested/needed
SUBGRAPH_N = int(os.environ.get("SUBGRAPH_N", "50"))
SUBGRAPH_SEED = int(os.environ.get("SUBGRAPH_SEED", "0"))
sampled = False
if isinstance(SUBGRAPH_N, int) and SUBGRAPH_N > 0 and G.number_of_nodes() > SUBGRAPH_N:
    rng = random.Random(SUBGRAPH_SEED)
    start = rng.choice(list(G.nodes))
    visited = set([start])
    q = deque([start])
    while q and len(visited) < SUBGRAPH_N:
        u = q.popleft()
        for v in G.neighbors(u):
            if v not in visited:
                visited.add(v)
                if len(visited) >= SUBGRAPH_N:
                    break
                q.append(v)
    # In case BFS didn't reach enough nodes due to disconnected components, pad with random nodes
    if len(visited) < SUBGRAPH_N:
        remaining = [n for n in G.nodes if n not in visited]
        rng.shuffle(remaining)
        for n in remaining[: max(0, SUBGRAPH_N - len(visited))]:
            visited.add(n)
    G = G.subgraph(list(visited)).copy()
    sampled = True
    print(f"Sampled subgraph with {len(visited)} nodes (target={SUBGRAPH_N}, seed={SUBGRAPH_SEED})")

# Ensure a categorical node_type exists for coloring
has_node_type_attr = any("node_type" in G.nodes[n] for n in G.nodes)
if not has_node_type_attr:
    if data is not None and hasattr(data, "node_type"):
        values = data.node_type.detach().cpu().view(-1).tolist()
        mapped = {}
        for i in range(len(values)):
            val = values[i]
            mapped[i] = "unknown" if val is None else str(val)
        nx.set_node_attributes(G, mapped, "node_type")
    elif data is not None and hasattr(data, "y"):
        values = data.y.detach().cpu().view(-1).tolist()
        mapped = {}
        for i in range(len(values)):
            val = values[i]
            label = f"class_{int(val)}" if val is not None else "class_unknown"
            mapped[i] = label
        nx.set_node_attributes(G, mapped, "node_type")

# If edge_kind missing but edge_type present, mirror it for stats display
for edge in G.edges(data=True):
    d = edge[2]
    if "edge_kind" not in d and "edge_type" in d:
        et = d["edge_type"]
        try:
            d["edge_kind"] = str(int(et))
        except Exception:
            d["edge_kind"] = str(et)

# Build a unified edgelist from payload if G edges lack attributes or are empty
node_keys = set(G.nodes)
node_is_tuple = any(isinstance(n, tuple) for n in node_keys)
edges_for_draw_all = []
if isinstance(obj, dict) and isinstance(obj.get("edge_index"), dict):
    for key_str, ei in obj["edge_index"].items():
        parts = str(key_str).split(":")
        if len(parts) != 3:
            continue
        src_t, _pid, dst_t = parts
        try:
            s_list = ei[0].tolist() if hasattr(ei, "tolist") else list(ei[0])
            d_list = ei[1].tolist() if hasattr(ei, "tolist") else list(ei[1])
        except Exception:
            continue
        for s_idx, d_idx in zip(s_list, d_list):
            try:
                si = int(s_idx)
                di = int(d_idx)
            except Exception:
                continue
            u = (src_t, si) if node_is_tuple else si
            v = (dst_t, di) if node_is_tuple else di
            if u in node_keys and v in node_keys:
                edges_for_draw_all.append((u, v))
else:
    # Fallback to whatever edges exist in G
    if G.is_multigraph():
        edges_for_draw_all = [(u, v) for u, v, _ in G.edges(keys=True)]
    else:
        edges_for_draw_all = list(G.edges())

# Layout (prefer layout driven by actual edges_for_draw_all)
G_for_layout = nx.Graph()
G_for_layout.add_nodes_from(G.nodes)
G_for_layout.add_edges_from(edges_for_draw_all)
pos = nx.spring_layout(G_for_layout, seed=0)

# Stats
num_nodes = G.number_of_nodes()
print(f"Number of nodes: {num_nodes}")
num_edges = G.number_of_edges()
print(f"Number of edges: {num_edges}")
node_type_counts = Counter(get_type(n) for n in G.nodes)
print(f"Node type counts: {node_type_counts}")
edge_kind_counts = Counter()
relation_counts = Counter()
print(f"Edge kind counts: {edge_kind_counts}")
print(f"Relation counts: {relation_counts}")
if G.is_multigraph():
    for u, v, k, d in G.edges(keys=True, data=True):
        edge_kind = d.get('edge_kind') or d.get('edge_type') or 'unknown'
        if isinstance(edge_kind, (int, float)):
            edge_kind = str(int(edge_kind))
        edge_kind_counts[edge_kind] += 1
        pid = d.get('property_id')
        if isinstance(pid, str):
            relation_counts[pid] += 1
else:
    for u, v, d in G.edges(data=True):
        edge_kind = d.get('edge_kind') or d.get('edge_type') or 'unknown'
        if isinstance(edge_kind, (int, float)):
            edge_kind = str(int(edge_kind))
        edge_kind_counts[edge_kind] += 1
        pid = d.get('property_id')
        if isinstance(pid, str):
            relation_counts[pid] += 1


print(f"Graph stats: nodes={num_nodes}, edges={num_edges}")
print(f"Node types: {dict(node_type_counts)}")
print(f"Edge kinds: {dict(edge_kind_counts)}")
print(f"Unique relation property_ids: {len(relation_counts)}")

# 1) Node colors by type
types = sorted({get_type(n) for n in G.nodes})
palette_nodes = [plt.cm.tab10(i % 10) for i in range(len(types))]
type_to_color = dict(zip(types, palette_nodes))

node_color = [type_to_color[get_type(n)] for n in G.nodes]

# 2) Edge colors by endpoint type-pair
pair_types = []
edgelist = edges_for_draw_all
for u, v in edgelist:
    tu, tv = get_type(u), get_type(v)
    pair_types.append(tuple(sorted((tu, tv))))
pair_types_unique = sorted(set(pair_types))

# assign colors to pairs; same-type uses that node type color; cross-type from a separate palette
cross_palette = cycle([plt.cm.tab20(i) for i in range(20)])
pair_to_color = {}
for a, b in pair_types_unique:
    if a == b:
        pair_to_color[(a, b)] = type_to_color[a]
    else:
        pair_to_color[(a, b)] = next(cross_palette)

edge_colors = [pair_to_color[tuple(sorted((get_type(u), get_type(v))))] for u, v in edgelist]
edges_for_draw = edgelist

# Draw
plt.figure(figsize=(9, 9))
nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=50)
nx.draw_networkx_edges(
    G, pos,
    edgelist=edges_for_draw,
    edge_color=edge_colors,
    width=1.2, alpha=0.95,
    arrows=False, connectionstyle='arc3,rad=0.08'
)
# Optional: node labels showing type
# nx.draw_networkx_labels(G, pos, labels={n: get_type(n) for n in G.nodes}, font_size=6)
plt.axis('off')
plt.title(f"Nodes: {num_nodes} | Edges: {num_edges} | Rel types: {len(relation_counts)}", fontsize=10)
os.makedirs(GRAPH_VIZ_DIR, exist_ok=True)
base = os.path.basename(PYG_GRAPH_PATH)
name_core = base[:-len(".pyg.pt")] if base.endswith(".pyg.pt") else os.path.splitext(base)[0]
name_core = f"{name_core}_sub{SUBGRAPH_N}" if sampled else name_core
outfile = os.path.join(GRAPH_VIZ_DIR, f"{name_core}_viz.png")
plt.tight_layout()
plt.savefig(outfile, dpi=300)
print(f"Saved visualization to {outfile}")
plt.show()

# === Per-relation subgraphs ===
REL_LANG = os.environ.get("REL_LANG", "en")
REL_TOP_K = int(os.environ.get("REL_TOP_K", "10"))
REL_ALL = os.environ.get("REL_ALL", "0") == "1"
REL_MAX_EDGES = int(os.environ.get("REL_MAX_EDGES", "1000"))

# Try to get human-readable labels for relations if present in loaded payload
relation_label_map = {}
if isinstance(obj, dict):
    key = f"relation_labels_{REL_LANG}"
    if key in obj and isinstance(obj[key], dict):
        relation_label_map = {str(k): str(v) for k, v in obj[key].items()}

def _safe_label(s):
    s = str(s)
    return s.replace("\n", " ").strip()

def _safe_fname(s):
    s = str(s)
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)[:80]

# Build property_id -> edge tuples present in sampled G
rel_to_edges = {}
if G.is_multigraph():
    for u, v, k, d in G.edges(keys=True, data=True):
        pid = d.get("property_id")
        if isinstance(pid, str):
            rel_to_edges.setdefault(pid, []).append((u, v, k))
else:
    for u, v, d in G.edges(data=True):
        pid = d.get("property_id")
        if isinstance(pid, str):
            rel_to_edges.setdefault(pid, []).append((u, v))

# Fallback: If property_id wasn't carried into G's edges, rebuild relation edges from payload
if not rel_to_edges and isinstance(obj, dict) and isinstance(obj.get("edge_index"), dict):
    G_nodes = set(G.nodes)
    for key_str, ei in obj["edge_index"].items():
        parts = str(key_str).split(":")
        if len(parts) != 3:
            continue
        src_t, pid, dst_t = parts
        # Convert to lists defensively
        try:
            s_list = ei[0].tolist() if hasattr(ei, "tolist") else list(ei[0])
            d_list = ei[1].tolist() if hasattr(ei, "tolist") else list(ei[1])
        except Exception:
            continue
        for s_idx, d_idx in zip(s_list, d_list):
            try:
                si = int(s_idx)
                di = int(d_idx)
            except Exception:
                continue
            u = (src_t, si)
            v = (dst_t, di)
            if u in G_nodes and v in G_nodes:
                # Use 2-tuples so this works for both Graph and MultiGraph edge_subgraph
                rel_to_edges.setdefault(pid, []).append((u, v))

# Order relations by frequency unless REL_ALL is set
rel_order = sorted(rel_to_edges.items(), key=lambda kv: len(kv[1]), reverse=True)
if not REL_ALL:
    rel_order = rel_order[:REL_TOP_K]

for i, (pid, edges) in enumerate(rel_order):
    if not edges:
        continue
    # Optionally sample edges to limit density
    if REL_MAX_EDGES > 0 and len(edges) > REL_MAX_EDGES:
        rng = random.Random(SUBGRAPH_SEED + i)
        edges = rng.sample(edges, REL_MAX_EDGES)
    # Build edge-induced subgraph for this relation
    G_rel = G.edge_subgraph(edges).copy()
    if G_rel.number_of_edges() == 0 or G_rel.number_of_nodes() == 0:
        continue
    # Layout and draw
    pos_rel = nx.spring_layout(G_rel, seed=SUBGRAPH_SEED)
    # Nodes colored by type as before
    types_rel = sorted({G_rel.nodes[n].get("node_type", "unknown") for n in G_rel.nodes})
    palette_nodes_rel = [plt.cm.tab10(i % 10) for i in range(len(types_rel))]
    type_to_color_rel = dict(zip(types_rel, palette_nodes_rel))
    node_color_rel = [type_to_color_rel[G_rel.nodes[n].get("node_type", "unknown")] for n in G_rel.nodes]
    edge_color_rel = plt.cm.tab20(i % 20)

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G_rel, pos_rel, node_color=node_color_rel, node_size=40)
    # Normalize edgelist for drawing (support both 2-tuple and 3-tuple forms)
    if edges and isinstance(edges[0], tuple) and len(edges[0]) == 3:
        edraw = [(u, v) for (u, v, _) in edges]
    else:
        edraw = list(edges)
    nx.draw_networkx_edges(
        G_rel, pos_rel,
        edgelist=edraw,
        edge_color=edge_color_rel, width=1.0, alpha=0.95,
        arrows=False, connectionstyle='arc3,rad=0.08'
    )
    plt.axis("off")
    label = relation_label_map.get(pid, pid)
    plt.title(f"{_safe_label(label)} ({pid}) | nodes={G_rel.number_of_nodes()} edges={G_rel.number_of_edges()}", fontsize=10)
    fname = os.path.join(GRAPH_VIZ_DIR, f"{name_core}_rel_{_safe_fname(pid)}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    print(f"Saved relation subgraph for {pid} -> {fname}")
    plt.close()

# === Entity vs Literal view (coolwarm like karate example) ===
# Map node types to community ids: entity -> 0, literal -> 1 (others -> 1)
color_map_comm = {"entity": 0, "literal": 1}
node_color_comm = [color_map_comm.get(G.nodes[n].get("node_type", "literal"), 1) for n in G.nodes()]
pos_comm = nx.spring_layout(G, seed=SUBGRAPH_SEED)
plt.figure(figsize=(8, 8))
nx.draw_networkx_nodes(
    G,
    pos=pos_comm,
    cmap=plt.get_cmap('coolwarm'),
    node_color=node_color_comm,
    node_size=50,
    alpha=0.95
)
nx.draw_networkx_edges(
    G,
    pos=pos_comm,
    edgelist=edges_for_draw_all,
    edge_color="#444",
    width=1.4,
    alpha=0.95,
    arrows=False,
    connectionstyle='arc3,rad=0.08'
)
plt.axis("off")
outfile_comm = os.path.join(GRAPH_VIZ_DIR, f"{name_core}_coolwarm.png")
plt.tight_layout()
plt.savefig(outfile_comm, dpi=300)
print(f"Saved entity-vs-literal view to {outfile_comm}")
plt.show()