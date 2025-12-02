#!/usr/bin/env python3
"""
Visualize a saved graph payload (.pt) from export_to_pt.py.

- Colors nodes by node type: entity, literal, external_id
- Colors edges by relation type (src_type:property_id:dst_type)
"""
from __future__ import annotations

import argparse
import random
from typing import Dict, Tuple, List, Optional, Iterable, Set

import matplotlib.pyplot as plt
import networkx as nx
import torch

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize .pt graph saved by export_to_pt.py")
    ap.add_argument("--pt", required=True, help="Path to saved .pt payload")
    ap.add_argument("--max_edges", type=int, default=5000, help="Maximum number of edges to draw (samples if larger)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for layout and sampling")
    ap.add_argument("--show_labels", action="store_true", help="Draw node labels (can be cluttered)")
    ap.add_argument("--only_rel", action="append", default=[], help="Limit to specific relation keys (repeatable). Example: entity:P31:entity")
    return ap.parse_args()

def reverse_index(id_to_idx: Dict[str, int]) -> Dict[int, str]:
    return {idx: _id for _id, idx in id_to_idx.items()}

def build_networkx_from_payload(
    payload: Dict[str, object],
    max_edges: int,
    only_rel: Optional[Set[str]],
    seed: int,
) -> Tuple[nx.MultiDiGraph, Dict[Tuple[str, int], str]]:
    """
    Returns:
      - MultiDiGraph with nodes keyed by (node_type, local_idx)
      - Labels dict for nodes (human-readable id for entities; synthetic for others)
    """
    node_index: Dict[str, Dict[str, int]] = payload["node_index"]  # type: ignore[assignment]
    edge_index: Dict[str, torch.Tensor] = payload["edge_index"]  # type: ignore[assignment]
    edge_claim_ids: Dict[str, List[Optional[str]]] = payload.get("edge_claim_ids", {})  # type: ignore[assignment]

    # Reverse map for nicer labels on entity nodes
    entity_rev = reverse_index(node_index.get("entity", {}))

    G = nx.MultiDiGraph()
    node_labels: Dict[Tuple[str, int], str] = {}

    def ensure_node(node_type: str, idx: int) -> None:
        key = (node_type, idx)
        if key in G:
            return
        G.add_node(key, node_type=node_type)
        if node_type == "entity":
            label = entity_rev.get(idx, f"entity:{idx}")
        else:
            label = f"{node_type}:{idx}"
        node_labels[key] = label

    # Collect all edges across relations, then sample if needed
    all_edges: List[Tuple[Tuple[str, int], Tuple[str, int], Dict[str, object]]] = []
    rng = random.Random(seed)

    for key_str, ei in edge_index.items():
        if only_rel and key_str not in only_rel:
            continue
        src_t, prop, dst_t = key_str.split(":")
        s_list = ei[0].tolist()
        d_list = ei[1].tolist()
        claims: List[Optional[str]] = edge_claim_ids.get(key_str, [None] * len(s_list))

        # Optionally downsample per relation to keep representation balanced
        rel_edges: List[Tuple[Tuple[str, int], Tuple[str, int], Dict[str, object]]] = []
        for s_idx, d_idx, cid in zip(s_list, d_list, claims):
            u = (src_t, int(s_idx))
            v = (dst_t, int(d_idx))
            rel_edges.append((u, v, {"edge_type": key_str, "property_id": prop, "claim_id": cid}))

        all_edges.extend(rel_edges)

    if len(all_edges) > max_edges:
        all_edges = rng.sample(all_edges, k=max_edges)

    # Add nodes and edges
    for (u_type, u_idx), (v_type, v_idx), attrs in all_edges:
        ensure_node(u_type, u_idx)
        ensure_node(v_type, v_idx)
        G.add_edge((u_type, u_idx), (v_type, v_idx), **attrs)

    return G, node_labels


def draw_graph(
    G: nx.MultiDiGraph,
    node_labels: Dict[Tuple[str, int], str],
    seed: int,
    show_labels: bool,
) -> None:
    # Node colors by type
    node_type_to_color = {
        "entity": plt.cm.tab10(0),
        "literal": plt.cm.tab10(2),
        "external_id": plt.cm.tab10(3),
    }
    default_node_color = plt.cm.tab10(9)
    node_colors = []
    for n in G.nodes:
        ntype = G.nodes[n].get("node_type", "unknown")
        node_colors.append(node_type_to_color.get(ntype, default_node_color))

    # Edge colors by relation key
    edge_types: List[str] = []
    edgelist: List[Tuple[Tuple[str, int], Tuple[str, int], int]] = []
    for u, v, k, d in G.edges(keys=True, data=True):
        edgelist.append((u, v, k))
        edge_types.append(d.get("edge_type", "unknown"))

    unique_edge_types = sorted(set(edge_types))
    palette = [plt.cm.tab20(i % 20) for i in range(max(1, len(unique_edge_types)))]
    edge_type_to_color = {t: palette[i % len(palette)] for i, t in enumerate(unique_edge_types)}
    edge_colors = [edge_type_to_color[t] for t in edge_types]

    pos = nx.spring_layout(G, seed=seed)

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=40, linewidths=0.0)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(u, v) for (u, v, _) in edgelist],
        edge_color=edge_colors,
        width=0.6,
        alpha=0.8,
        arrows=False,
        connectionstyle="arc3,rad=0.06",
    )
    if show_labels:
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=6)

    # Legend (node types)
    from matplotlib.lines import Line2D
    node_legend = [
        Line2D([0], [0], marker='o', color='w', label=ntype, markerfacecolor=col, markersize=8)
        for ntype, col in node_type_to_color.items()
    ]
    plt.legend(handles=node_legend, loc="upper right", fontsize=8, frameon=False, title="Node types")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main() -> int:
    args = parse_args()
    payload = torch.load(args.pt, map_location="cpu")
    only_rel = set(args.only_rel) if args.only_rel else None
    G, node_labels = build_networkx_from_payload(payload, max_edges=args.max_edges, only_rel=only_rel, seed=args.seed)
    draw_graph(G, node_labels, seed=args.seed, show_labels=args.show_labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


