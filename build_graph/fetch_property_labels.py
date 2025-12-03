#!/usr/bin/env python3
"""
CLI: Fetch language-specific labels for Wikidata property IDs and save to JSON.

Usage examples:
  python fetch_property_labels.py --pids P170,P31,P279 --lang en --out prop_labels_en.json
  python fetch_property_labels.py --from_pt /path/graph.pt --lang en --out prop_labels_en.json
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import torch

# Robust import: allow running from repo root or the script directory
try:
    from utils.wikidata_labels import (
        fetch_property_labels,
        normalize_property_ids,
        save_json,
    )
except ModuleNotFoundError:
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from utils.wikidata_labels import (
        fetch_property_labels,
        normalize_property_ids,
        save_json,
    )

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fetch Wikidata property labels to JSON")
    ap.add_argument("--pids", default="", help="Comma-separated property IDs, e.g., P170,P31")
    ap.add_argument("--from_pt", default="", help="Path to graph .pt; extracts property IDs from edge_index keys")
    ap.add_argument("--lang", default="en", help="Language code (e.g., en, de)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    # New: rewrite a cleaned graph payload by adding language-specific relation labels
    ap.add_argument("--graph_in", default="", help="Path to input cleaned graph .pt (dict payload expected)")
    ap.add_argument("--graph_out", default="", help="Path to output graph .pt with language-specific labels added")
    # New: attach node textual data (from entity metadata) and build label->text pairs
    ap.add_argument("--entity_meta", default="", help="Path to entity_metadata_{lang}.pt (for entity text)")
    ap.add_argument("--edge_text_max", type=int, default=1000, help="Max text pairs stored per labeled relation")
    return ap.parse_args()

def collect_pids_from_pt(path: str) -> List[str]:
    p = torch.load(path, map_location="cpu")
    edge_index: Dict[str, torch.Tensor] = p.get("edge_index", {})
    pids: List[str] = []
    for key in edge_index.keys():
        parts = key.split(":")
        if len(parts) == 3:
            _, pid, _ = parts
            pids.append(pid)
    return normalize_property_ids(pids)

def main() -> int:
    args = parse_args()
    pids: List[str] = []
    if args.pids:
        pids.extend([s.strip() for s in args.pids.split(",") if s.strip()])
    if args.from_pt:
        pids.extend(collect_pids_from_pt(args.from_pt))
    pids = normalize_property_ids(pids)
    labels = fetch_property_labels(pids, lang=args.lang)
    save_json(labels, args.out)
    print(f"Saved {len(labels)} labels to {args.out}")

    # If requested, load a cleaned graph payload and attach language-specific labels
    if args.graph_in:
        payload = torch.load(args.graph_in, map_location="cpu")
        if not isinstance(payload, dict) or "edge_index" not in payload:
            raise TypeError("graph_in must be a dict payload with 'edge_index'")

        # Collect all property IDs present in the graph
        rel_keys = list(payload["edge_index"].keys())  # type: ignore[index]
        rel_pids = []
        for k in rel_keys:
            parts = str(k).split(":")
            if len(parts) == 3:
                rel_pids.append(parts[1])
        rel_pids = normalize_property_ids(rel_pids)

        # Merge with any pre-fetched labels; fetch missing ones
        have = set(labels.keys())
        need = [pid for pid in rel_pids if pid not in have]
        if need:
            extra = fetch_property_labels(need, lang=args.lang)
            labels.update(extra)

        # Helper to sanitize label for use inside keys (avoid colons)
        def sanitize_label(s: str) -> str:
            s = s.replace(":", " - ")
            s = s.replace("\n", " ").strip()
            return s if s else "unknown"

        edge_index = payload["edge_index"]  # type: ignore[assignment]
        edge_claim_ids = payload.get("edge_claim_ids", {})

        # Build mappings:
        # - property_labels_{lang}: pid -> label
        # - relation_labels_{lang}: "src:pid:dst" -> label
        # - edge_index_label_{lang}: "src:label:dst" -> Tensor[2, E] (merged per label)
        # - edge_text_pairs_label_{lang}: "src:label:dst" -> [[src_text, dst_text], ...]
        prop_key = f"property_labels_{args.lang}"
        rel_key = f"relation_labels_{args.lang}"
        ei_label_key = f"edge_index_label_{args.lang}"
        eci_label_key = f"edge_claim_ids_label_{args.lang}"
        et_pairs_key = f"edge_text_pairs_label_{args.lang}"

        payload[prop_key] = {pid: labels.get(pid, pid) for pid in rel_pids}

        relation_labels = {}
        for k in rel_keys:
            parts = str(k).split(":")
            if len(parts) == 3:
                _, pid, _ = parts
                relation_labels[k] = labels.get(pid, pid)
        payload[rel_key] = relation_labels

        # Aggregate edges by label-key
        import torch as _torch  # local alias to avoid confusion with utils imports
        edge_index_label = {}
        edge_claim_ids_label = {}
        edge_text_pairs_label = {}

        # Optional: load entity metadata to attach textual data
        entity_text_by_index = {}
        literal_idx_to_text = {}
        external_idx_to_text = {}
        if args.entity_meta:
            meta = torch.load(args.entity_meta, map_location="cpu")
            if isinstance(meta, dict):
                # Determine which text key to use
                meta_lang = meta.get("lang", args.lang)
                # If available, merge entity metadata maps into the output payload
                lbl_key = f"entity_labels_{meta_lang}"
                desc_key = f"entity_descriptions_{meta_lang}"
                alias_key = f"entity_aliases_{meta_lang}"
                if lbl_key in meta and isinstance(meta[lbl_key], dict):
                    payload[lbl_key] = meta[lbl_key]
                if desc_key in meta and isinstance(meta[desc_key], dict):
                    payload[desc_key] = meta[desc_key]
                if alias_key in meta and isinstance(meta[alias_key], dict):
                    payload[alias_key] = meta[alias_key]
                preferred_key = f"entity_text_{meta_lang}_by_index"
                if preferred_key in meta and isinstance(meta[preferred_key], dict):
                    entity_text_by_index = {int(k): str(v) for k, v in meta[preferred_key].items()}
                    # Also embed into payload so downstream consumers can use it directly
                    payload[preferred_key] = entity_text_by_index
                else:
                    # Fallback: any entity_text_*_by_index key
                    for mk, mv in meta.items():
                        if isinstance(mk, str) and mk.startswith("entity_text_") and mk.endswith("_by_index") and isinstance(mv, dict):
                            entity_text_by_index = {int(k): str(v) for k, v in mv.items()}
                            # Store under detected key
                            payload[mk] = entity_text_by_index
                            break

        # Invert node_index for literal/external_id to derive simple text if available
        node_index = payload.get("node_index", {})
        if isinstance(node_index, dict):
            if "literal" in node_index and isinstance(node_index["literal"], dict):
                for lit_id, lit_idx in node_index["literal"].items():
                    try:
                        literal_idx_to_text[int(lit_idx)] = str(lit_id)
                    except Exception:
                        continue
            if "external_id" in node_index and isinstance(node_index["external_id"], dict):
                for ex_id, ex_idx in node_index["external_id"].items():
                    try:
                        external_idx_to_text[int(ex_idx)] = str(ex_id)
                    except Exception:
                        continue
        for k, ei in edge_index.items():  # type: ignore[union-attr]
            parts = str(k).split(":")
            if len(parts) != 3:
                continue
            src_t, pid, dst_t = parts
            label = sanitize_label(labels.get(pid, pid))
            new_key = f"{src_t}:{label}:{dst_t}"
            # convert to lists
            if hasattr(ei, "tolist"):
                src_list = ei[0].tolist()
                dst_list = ei[1].tolist()
            else:
                src_list = list(ei[0])
                dst_list = list(ei[1])

            # Append/merge
            if new_key in edge_index_label:
                prev = edge_index_label[new_key]
                if hasattr(prev, "tolist"):
                    ps, pd = prev[0].tolist(), prev[1].tolist()
                else:
                    ps, pd = list(prev[0]), list(prev[1])
                ps.extend(src_list)
                pd.extend(dst_list)
                edge_index_label[new_key] = _torch.tensor([ps, pd], dtype=_torch.long)
            else:
                edge_index_label[new_key] = _torch.tensor([src_list, dst_list], dtype=_torch.long)

            # Claims if present
            claims = edge_claim_ids.get(k)
            if claims is not None:
                if new_key in edge_claim_ids_label:
                    edge_claim_ids_label[new_key].extend(claims)
                else:
                    edge_claim_ids_label[new_key] = list(claims)

            # Text pairs if requested via entity_meta
            if args.entity_meta:
                pairs = edge_text_pairs_label.get(new_key)
                if pairs is None:
                    pairs = []
                    edge_text_pairs_label[new_key] = pairs
                # Build text for endpoints by type
                for s_idx, d_idx in zip(src_list, dst_list):
                    if args.edge_text_max > 0 and len(pairs) >= args.edge_text_max:
                        break
                    try:
                        si = int(s_idx)
                        di = int(d_idx)
                    except Exception:
                        continue
                    # src
                    if src_t == "entity":
                        s_text = entity_text_by_index.get(si, "")
                    elif src_t == "literal":
                        s_text = literal_idx_to_text.get(si, f"literal:{si}")
                    elif src_t == "external_id":
                        s_text = external_idx_to_text.get(si, f"external_id:{si}")
                    else:
                        s_text = f"{src_t}:{si}"
                    # dst
                    if dst_t == "entity":
                        d_text = entity_text_by_index.get(di, "")
                    elif dst_t == "literal":
                        d_text = literal_idx_to_text.get(di, f"literal:{di}")
                    elif dst_t == "external_id":
                        d_text = external_idx_to_text.get(di, f"external_id:{di}")
                    else:
                        d_text = f"{dst_t}:{di}"
                    pairs.append([s_text, d_text])

        payload[ei_label_key] = edge_index_label
        if edge_claim_ids_label:
            payload[eci_label_key] = edge_claim_ids_label
        if args.entity_meta and edge_text_pairs_label:
            payload[et_pairs_key] = edge_text_pairs_label

        # Decide output
        if not args.graph_out:
            import os as _os
            base = _os.path.basename(args.graph_in)
            stem = base[:-3] if base.endswith(".pt") else base
            args.graph_out = _os.path.join(_os.path.dirname(args.graph_in), f"{stem}_{args.lang}_labeled.pt")
        torch.save(payload, args.graph_out)
        print(f"Saved labeled graph to {args.graph_out}")
    return 0 

if __name__ == "__main__":
    raise SystemExit(main())



