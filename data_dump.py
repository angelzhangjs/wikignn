#!/usr/bin/env python3
"""
Dump a small, artwork-only subset from a simple-wikidata-db staged directory.

Overview:
- Computes subclass-of (P279) closure starting from work of art (Q838948).
- Finds entities with instance of (P31) in that class set.
- Joins basic metadata from labels, descriptions, aliases, wikipedia_links.
- Writes compact JSONL rows, with --limit to control subset size.

Requirements:
- You must have already run simple-wikidata-db's preprocess_dump.py with a language (e.g., en).
  The staged directory should contain subdirectories: entity_rels, labels, descriptions, aliases, wikipedia_links
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from typing import Dict, Iterator, List, Optional, Set


DEFAULT_ROOT_ART_QID = "Q838948"  # work of art
P_INSTANCE_OF = "P31"
P_SUBCLASS_OF = "P279"

def iter_jsonl_files(table_dir: str) -> Iterator[str]:
    """Yield JSONL file paths under a table directory."""
    if not os.path.isdir(table_dir):
        return
    # Common patterns created by simple-wikidata-db
    patterns = ["*.jsonl", "*.json"]
    for pattern in patterns:
        for path in sorted(glob(os.path.join(table_dir, pattern))):
            yield path

def iter_rows_from_table(db_dir: str, table_name: str) -> Iterator[Dict]:
    """Stream rows from a table directory as parsed JSON objects."""
    table_dir = os.path.join(db_dir, table_name)
    for fp in iter_jsonl_files(table_dir):
        try:
            with open(fp, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        # Skip malformed lines
                        continue
        except FileNotFoundError:
            continue

def build_art_class_closure(db_dir: str, root_qid: str, max_passes: int = 10) -> Set[str]:
    """
    Compute transitive closure of subclasses of root_qid using entity_rels (P279).
    Performs multiple passes over entity_rels until no growth or max_passes reached.
    """
    art_classes: Set[str] = {root_qid}
    passes = 0
    while passes < max_passes:
        passes += 1
        discovered_this_pass = 0
        for row in iter_rows_from_table(db_dir, "entity_rels"):
            property_id = row.get("property_id")
            if property_id != P_SUBCLASS_OF:
                continue
            parent_qid = row.get("value")
            child_qid = row.get("qid")
            if not parent_qid or not child_qid:
                continue
            if parent_qid in art_classes and child_qid not in art_classes:
                art_classes.add(child_qid)
                discovered_this_pass += 1
        if discovered_this_pass == 0:
            break
    return art_classes

def collect_artwork_qids(db_dir: str, art_class_qids: Set[str], limit: Optional[int]) -> Set[str]:
    """
    Find entities that have P31 pointing to any class in art_class_qids.
    Stops early if limit is reached (if provided).
    """
    results: Set[str] = set()
    if limit is not None and limit <= 0:
        return results

    for row in iter_rows_from_table(db_dir, "entity_rels"):
        if row.get("property_id") != P_INSTANCE_OF:
            continue
        qid = row.get("qid")
        value_qid = row.get("value")
        if not qid or not value_qid:
            continue
        if value_qid in art_class_qids:
            results.add(qid)
            if limit is not None and len(results) >= limit:
                break
    return results

def build_label_map(db_dir: str, target_qids: Set[str]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for row in iter_rows_from_table(db_dir, "labels"):
        qid = row.get("qid")
        if qid in target_qids:
            label = row.get("label")
            if isinstance(label, str):
                labels[qid] = label
    return labels


def build_description_map(db_dir: str, target_qids: Set[str]) -> Dict[str, str]:
    descs: Dict[str, str] = {}
    for row in iter_rows_from_table(db_dir, "descriptions"):
        qid = row.get("qid")
        if qid in target_qids:
            description = row.get("description")
            if isinstance(description, str):
                descs[qid] = description
    return descs


def build_aliases_map(db_dir: str, target_qids: Set[str]) -> Dict[str, List[str]]:
    aliases_map: Dict[str, List[str]] = {}
    for row in iter_rows_from_table(db_dir, "aliases"):
        qid = row.get("qid")
        if qid in target_qids:
            alias = row.get("alias")
            if isinstance(alias, str):
                aliases_map.setdefault(qid, []).append(alias)
    return aliases_map


def build_wikipedia_map(db_dir: str, target_qids: Set[str]) -> Dict[str, str]:
    wiki_map: Dict[str, str] = {}
    for row in iter_rows_from_table(db_dir, "wikipedia_links"):
        qid = row.get("qid")
        if qid in target_qids:
            title = row.get("wiki_title")
            if isinstance(title, str):
                wiki_map[qid] = title
    return wiki_map


def write_output(
    out_path: str,
    qids: Set[str],
    labels: Dict[str, str],
    descriptions: Dict[str, str],
    aliases: Dict[str, List[str]],
    wiki_titles: Dict[str, str],
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for qid in sorted(qids):
            row = {
                "qid": qid,
                "label": labels.get(qid),
                "description": descriptions.get(qid),
                "aliases": aliases.get(qid, []),
                "wikipedia_title": wiki_titles.get(qid),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def collect_outgoing_entity_rels(db_dir: str, source_qids: Set[str]) -> Iterator[Dict]:
    for row in iter_rows_from_table(db_dir, "entity_rels"):
        qid = row.get("qid")
        if qid in source_qids:
            yield {
                "direction": "out",
                "claim_id": row.get("claim_id"),
                "source": qid,
                "property_id": row.get("property_id"),
                "target": row.get("value"),
            }

def collect_incoming_entity_rels(db_dir: str, target_qids: Set[str]) -> Iterator[Dict]:
    for row in iter_rows_from_table(db_dir, "entity_rels"):
        value_qid = row.get("value")
        if value_qid in target_qids:
            yield {
                "direction": "in",
                "claim_id": row.get("claim_id"),
                "source": row.get("qid"),
                "property_id": row.get("property_id"),
                "target": value_qid,
            }

def collect_entity_values(db_dir: str, source_qids: Set[str]) -> Iterator[Dict]:
    for row in iter_rows_from_table(db_dir, "entity_values"):
        qid = row.get("qid")
        if qid in source_qids:
            yield {
                "claim_id": row.get("claim_id"),
                "source": qid,
                "property_id": row.get("property_id"),
                "value": row.get("value"),
            }

def collect_external_ids(db_dir: str, source_qids: Set[str]) -> Iterator[Dict]:
    for row in iter_rows_from_table(db_dir, "external_ids"):
        qid = row.get("qid")
        if qid in source_qids:
            yield {
                "claim_id": row.get("claim_id"),
                "source": qid,
                "property_id": row.get("property_id"),
                "value": row.get("value"),
            }

def write_jsonl(path: str, rows: Iterator[Dict]) -> int:
    ensure_dir(os.path.dirname(path) or ".")
    count = 0
    with open(path, "w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count

def build_nodes_and_edges(
    db_dir: str,
    artwork_qids: Set[str],
    include_incoming_rels: bool,
    graph_out_dir: str,
) -> Dict[str, int]:
    stats: Dict[str, int] = {}

    # Outgoing edges from artworks
    out_rels_iter = collect_outgoing_entity_rels(db_dir, artwork_qids)
    rels_out_path = os.path.join(graph_out_dir, "edges_entity_rels_outgoing.jsonl")
    stats["edges_entity_rels_outgoing"] = write_jsonl(rels_out_path, out_rels_iter)

    # Optional incoming edges to artworks
    rels_in_count = 0
    rels_in_path = os.path.join(graph_out_dir, "edges_entity_rels_incoming.jsonl")
    if include_incoming_rels:
        in_rels_iter = collect_incoming_entity_rels(db_dir, artwork_qids)
        rels_in_count = write_jsonl(rels_in_path, in_rels_iter)
    stats["edges_entity_rels_incoming"] = rels_in_count

    # Attribute edges from artworks
    values_iter = collect_entity_values(db_dir, artwork_qids)
    values_path = os.path.join(graph_out_dir, "edges_entity_values.jsonl")
    stats["edges_entity_values"] = write_jsonl(values_path, values_iter)

    external_ids_iter = collect_external_ids(db_dir, artwork_qids)
    external_ids_path = os.path.join(graph_out_dir, "edges_external_ids.jsonl")
    stats["edges_external_ids"] = write_jsonl(external_ids_path, external_ids_iter)

    # Gather claim_ids from the produced edge files to fetch qualifiers
    claim_ids: Set[str] = set()
    def add_claim_ids_from(path: str) -> None:
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                cid = obj.get("claim_id")
                if isinstance(cid, str):
                    claim_ids.add(cid)
                    
    add_claim_ids_from(rels_out_path)
    if include_incoming_rels:
        add_claim_ids_from(rels_in_path)
        
    add_claim_ids_from(values_path)
    add_claim_ids_from(external_ids_path)

    qualifiers_path = os.path.join(graph_out_dir, "qualifiers.jsonl")
    def iter_qualifiers(db_dir: str, target_claim_ids: Set[str]) -> Iterator[Dict]:
        for row in iter_rows_from_table(db_dir, "qualifiers"):
            cid = row.get("claim_id")
            if cid in target_claim_ids:
                yield row
    stats["qualifiers"] = write_jsonl(qualifiers_path, iter_qualifiers(db_dir, claim_ids))

    # Build node set = artworks U neighbors referenced in rel edges
    node_qids: Set[str] = set(artwork_qids)
    def add_nodes_from_edge_file(path: str) -> None:
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                src = obj.get("source")
                tgt = obj.get("target")
                if isinstance(src, str) and src.startswith("Q"):
                    node_qids.add(src)
                if isinstance(tgt, str) and tgt.startswith("Q"):
                    node_qids.add(tgt)
                    
    add_nodes_from_edge_file(rels_out_path)
    if include_incoming_rels:
        add_nodes_from_edge_file(rels_in_path)

    # Write node metadata
    labels = build_label_map(db_dir, node_qids)
    descriptions = build_description_map(db_dir, node_qids)
    aliases = build_aliases_map(db_dir, node_qids)
    wiki_titles = build_wikipedia_map(db_dir, node_qids)

    nodes_out_path = os.path.join(graph_out_dir, "nodes.jsonl")
    write_output(
        out_path=nodes_out_path,
        qids=node_qids,
        labels=labels,
        descriptions=descriptions,
        aliases=aliases,
        wiki_titles=wiki_titles,
    )
    stats["nodes"] = len(node_qids)

    # Write a small meta.json with counts
    meta_path = os.path.join(graph_out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as fmeta:
        json.dump(stats, fmeta, ensure_ascii=False, indent=2)
    return stats

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump an artwork-only subset or subgraph from simple-wikidata-db staged data."
    )
    parser.add_argument(
        "--db_dir",
        required=True,
        help="Path to simple-wikidata-db out_dir (where tables were written).",
    )
    parser.add_argument(
        "--out",
        default="artworks_subset.jsonl",
        help="Output JSONL path for simple node listing (default: artworks_subset.jsonl).",
    )
    parser.add_argument(
        "--graph_out_dir",
        default=None,
        help="If provided, writes a subgraph to this directory: nodes and edges JSONL files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Maximum number of artworks to dump (default: 5000).",
    )
    parser.add_argument(
        "--root_art_qid",
        default=DEFAULT_ROOT_ART_QID,
        help="Root class QID to treat as 'work of art' (default: Q838948).",
    )
    parser.add_argument(
        "--max_passes",
        type=int,
        default=8,
        help="Max passes for subclass-of (P279) closure.",
    )
    parser.add_argument(
        "--include_incoming_rels",
        action="store_true",
        help="Also include incoming entity_rels edges (edges whose target is an artwork).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    db_dir = args.db_dir
    if not os.path.isdir(db_dir):
        print(f"db_dir not found: {db_dir}", file=sys.stderr)
        return 2

    # 1) Subclass-of closure
    art_class_qids = build_art_class_closure(
        db_dir=db_dir, root_qid=args.root_art_qid, max_passes=args.max_passes
    )

    # 2) Collect artwork QIDs (P31 in art_class_qids)
    artwork_qids = collect_artwork_qids(
        db_dir=db_dir, art_class_qids=art_class_qids, limit=args.limit
    )
    if not artwork_qids:
        print("No artworks found with the given criteria.", file=sys.stderr)
        return 0

    # 3) Join basic metadata
    labels = build_label_map(db_dir, artwork_qids)
    descriptions = build_description_map(db_dir, artwork_qids)
    aliases = build_aliases_map(db_dir, artwork_qids)
    wiki_titles = build_wikipedia_map(db_dir, artwork_qids)

    # 4) Write outputs
    # 4a) Always write a compact listing for artworks (for convenience)
    write_output(out_path=args.out, qids=artwork_qids, labels=labels, descriptions=descriptions, aliases=aliases, wiki_titles=wiki_titles)
    print(f"Wrote {len(artwork_qids)} artworks to {args.out}")

    # 4b) Optionally build a graph dump
    if args.graph_out_dir:
        ensure_dir(args.graph_out_dir)
        stats = build_nodes_and_edges(
            db_dir=db_dir,
            artwork_qids=artwork_qids,
            include_incoming_rels=bool(args.include_incoming_rels),
            graph_out_dir=args.graph_out_dir,
        )
        print(f"Wrote artwork subgraph to {args.graph_out_dir}: {json.dumps(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


