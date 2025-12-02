import os, json
import networkx as nx

def _iter_jsonl(path):
    if not path or not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_artwork_graph(graph_out_dir: str) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    claim_to_edge = {}  # claim_id -> (u, v, key)

    # 1) Nodes (QIDs) with metadata
    for row in _iter_jsonl(os.path.join(graph_out_dir, "nodes.jsonl")):
        qid = row["qid"]
        G.add_node(
            qid,
            node_kind="entity",
            label=row.get("label"),
            description=row.get("description"),
            aliases=row.get("aliases", []),
            wikipedia_title=row.get("wikipedia_title"),
        )

    # 2) Q->Q edges (entity_rels)
    for fname, direction in [
        ("edges_entity_rels_outgoing.jsonl", "out"),
        ("edges_entity_rels_incoming.jsonl", "in"),
    ]:
        path = os.path.join(graph_out_dir, fname)
        for row in _iter_jsonl(path):
            u = row["source"]
            v = row["target"]
            cid = row["claim_id"]
            ekey = cid or f"{row.get('property_id','')}-{u}->{v}"
            G.add_edge(
                u, v, key=ekey,
                edge_kind="entity_rel",
                property_id=row.get("property_id"),
                claim_id=cid,
                direction=direction,
            )
            if cid:
                claim_to_edge[cid] = (u, v, ekey)

    # 3) Q->literal edges (entity_values)
    for row in _iter_jsonl(os.path.join(graph_out_dir, "edges_entity_values.jsonl")):
        u = row["source"]
        cid = row["claim_id"]
        prop = row.get("property_id")
        lit_value = row.get("value")
        lit_id = f"lit:{prop}:{lit_value}"
        G.add_node(lit_id, node_kind="literal", literal_value=lit_value, property_id=prop)
        G.add_edge(
            u, lit_id, key=cid or lit_id,
            edge_kind="entity_value",
            property_id=prop,
            claim_id=cid,
        )
        if cid:
            claim_to_edge[cid] = (u, lit_id, cid or lit_id)

    # 4) Q->identifier edges (external_ids)
    for row in _iter_jsonl(os.path.join(graph_out_dir, "edges_external_ids.jsonl")):
        u = row["source"]
        cid = row["claim_id"]
        prop = row.get("property_id")
        ext_val = row.get("value")
        ext_id = f"ext:{prop}:{ext_val}"
        G.add_node(ext_id, node_kind="external_id", external_id=ext_val, property_id=prop)
        G.add_edge(
            u, ext_id, key=cid or ext_id,
            edge_kind="external_id",
            property_id=prop,
            claim_id=cid,
        )
        if cid:
            claim_to_edge[cid] = (u, ext_id, cid or ext_id)

    # 5) Qualifiers (attach to edge by claim_id)
    for row in _iter_jsonl(os.path.join(graph_out_dir, "qualifiers.jsonl")):
        cid = row.get("claim_id")
        if not cid or cid not in claim_to_edge:
            continue
        u, v, k = claim_to_edge[cid]
        data = G[u][v][k]
        quals = data.get("qualifiers")
        if not isinstance(quals, list):
            quals = []
            data["qualifiers"] = quals
        quals.append({
            "qualifier_id": row.get("qualifier_id"),
            "property_id": row.get("property_id"),
            "value": row.get("value"),
        })

    return G

