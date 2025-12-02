#!/usr/bin/env python3
"""
Fetch entity labels/descriptions/aliases for Q-ids in a cleaned graph payload
and save them as a .pt file for downstream CLIP text embedding.

Examples:
  python fetch_entity_metadata.py \
    --graph_in /home/ghr/angel/gnn/graph_output/clean_graph.pyg.pt \
    --lang en
"""
from __future__ import annotations
import argparse
import os
from typing import Dict, List, Tuple, Any
import torch

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fetch Wikidata entity metadata (labels/descriptions/aliases) for graph nodes.")
    ap.add_argument("--graph_in", default="/home/ghr/angel/gnn/graph_output/clean_graph.pyg.pt", help="Path to cleaned graph payload (.pt)")
    ap.add_argument("--lang", default="en", help="Language code (e.g., en, de)")
    ap.add_argument("--out", default="", help="Output .pt path (default: graph_output/entity_metadata_{lang}.pt)")
    ap.add_argument("--batch_size", type=int, default=50, help="Max QIDs per request (Wikidata API is fine with 50)")
    ap.add_argument("--timeout", type=int, default=20, help="HTTP timeout per request (seconds)")
    ap.add_argument("--max_retries", type=int, default=4, help="Max retries per request")
    ap.add_argument("--retry_backoff", type=float, default=1.5, help="Base backoff seconds (exponential)")
    ap.add_argument("--print_summary", action="store_true", help="Print a brief summary and a few example nodes")
    ap.add_argument("--print_all", action="store_true", help="Print all nodes with meta text (can be very long)")
    ap.add_argument("--max_print", type=int, default=20, help="Max examples to print when --print_summary (ignored if --print_all)")
    return ap.parse_args()

def normalize_qids(qids: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for q in qids:
        if not isinstance(q, str):
            continue
        q = q.strip()
        if not q or not q.startswith("Q"):
            continue
        num = q[1:]
        if not num.isdigit():
            continue
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out

def fetch_entities(
    qids: List[str],
    lang: str,
    timeout: int = 20,
    batch_size: int = 50,
    max_retries: int = 4,
    retry_backoff: float = 1.5,
) -> Dict[str, Dict[str, Any]]:
    """
    Returns mapping Qid -> {label, description, aliases}
    """
    try:
        import requests  # type: ignore
    except Exception as e:
        raise RuntimeError("The 'requests' package is required. Install with: pip install requests") from e

    import time

    qids = normalize_qids(qids)
    result: Dict[str, Dict[str, Any]] = {}
    if not qids:
        return result
    headers = {
        "User-Agent": "angel-gnn/0.1 (+https://example.org; data collection for research; contact: user@example.org)"
    }
    for i in range(0, len(qids), max(1, int(batch_size))):
        chunk = qids[i : i + max(1, int(batch_size))]
        j = None
        for attempt in range(max(1, int(max_retries))):
            try:
                r = requests.get(
                    "https://www.wikidata.org/w/api.php",
                    params={
                        "action": "wbgetentities",
                        "ids": "|".join(chunk),
                        "props": "labels|descriptions|aliases",
                        "languages": lang,
                        "format": "json",
                        "formatversion": "2",
                    },
                    headers=headers,
                    timeout=timeout,
                )
                if r.status_code == 429:
                    # rate limited; respect Retry-After if present
                    ra = r.headers.get("Retry-After")
                    sleep_s = float(ra) if ra and ra.isdigit() else (retry_backoff * (2 ** attempt))
                    time.sleep(sleep_s)
                    continue
                r.raise_for_status()
                j = r.json()
                break
            except Exception:
                if attempt == max(1, int(max_retries)) - 1:
                    # Give up on this chunk; move on
                    j = None
                else:
                    time.sleep(retry_backoff * (2 ** attempt))
        if not isinstance(j, dict):
            continue
        ents = j.get("entities", {}) if isinstance(j, dict) else {}
        for qid, ent in ents.items():
            if not isinstance(ent, dict):
                continue
            # formatversion=2 returns flattened language objects
            label = ent.get("labels", {}).get(lang) if isinstance(ent.get("labels"), dict) else None
            if isinstance(label, dict):
                label = label.get("value")
            desc = ent.get("descriptions", {}).get(lang) if isinstance(ent.get("descriptions"), dict) else None
            if isinstance(desc, dict):
                desc = desc.get("value")
            aliases = ent.get("aliases", {}).get(lang, []) if isinstance(ent.get("aliases"), dict) else []
            aliases_out: List[str] = []
            if isinstance(aliases, list):
                for a in aliases:
                    if isinstance(a, dict):
                        val = a.get("value")
                        if isinstance(val, str) and val.strip():
                            aliases_out.append(val.strip())
            result[qid] = {
                "label": label if isinstance(label, str) and label.strip() else "",
                "description": desc if isinstance(desc, str) and desc.strip() else "",
                "aliases": aliases_out,
            }
    return result

def invert_mapping(id_to_idx: Dict[str, int]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for _id, idx in id_to_idx.items():
        try:
            out[int(idx)] = str(_id)
        except Exception:
            continue
    return out

def make_text_by_index(qid_to_idx: Dict[str, int], meta: Dict[str, Dict[str, Any]]) -> Dict[int, str]:
    """
    For CLIP text embedding convenience, build an index->text mapping using
    "Label. Description" if available, else label, else empty string.
    """
    idx_to_qid = invert_mapping(qid_to_idx)
    out: Dict[int, str] = {}
    for idx, qid in idx_to_qid.items():
        m = meta.get(qid, {})
        label = m.get("label", "") or ""
        desc = m.get("description", "") or ""
        text = f"{label}. {desc}".strip().strip(".")
        out[idx] = text if text else label
    return out

def main() -> int:
    args = parse_args()
    payload = torch.load(args.graph_in, map_location="cpu")
    if not isinstance(payload, dict) or "node_index" not in payload:
        raise TypeError("graph_in must be a dict payload with 'node_index'")
    node_index = payload["node_index"]
    if not isinstance(node_index, dict) or "entity" not in node_index:
        raise TypeError("graph_in payload does not contain 'node_index[\"entity\"]'")
    qid_to_idx: Dict[str, int] = node_index["entity"]

    qids = normalize_qids(list(qid_to_idx.keys()))
    meta = fetch_entities(
        qids,
        lang=args.lang,
        timeout=args.timeout,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
    )

    # Build language-specific outputs
    labels = {qid: (m.get("label") or "") for qid, m in meta.items()}
    descriptions = {qid: (m.get("description") or "") for qid, m in meta.items()}
    aliases = {qid: (m.get("aliases") or []) for qid, m in meta.items()}
    text_by_index = make_text_by_index(qid_to_idx, meta)

    if not args.out:
        out_dir = "/home/ghr/angel/gnn/graph_output"
        os.makedirs(out_dir, exist_ok=True)
        args.out = os.path.join(out_dir, f"entity_metadata_{args.lang}.pt")

    torch.save(
        {
            "lang": args.lang,
            f"entity_labels_{args.lang}": labels,
            f"entity_descriptions_{args.lang}": descriptions,
            f"entity_aliases_{args.lang}": aliases,
            f"entity_text_{args.lang}_by_index": text_by_index,
        },
        args.out,
    )
    print(f"Saved entity metadata to: {args.out}")
    print(f"Entities: {len(qids)} | Labeled: {len(labels)}")

    # Optional summaries/prints
    if args.print_summary or args.print_all:
        print("\n== Entity index summary ==")
        print(f"Total entities in node_index['entity']: {len(qid_to_idx)}")
        print(f"Language: {args.lang}")
        idx_to_qid = invert_mapping(qid_to_idx)
        # Build rows sorted by index for readability
        rows = []
        for idx in sorted(idx_to_qid.keys()):
            qid = idx_to_qid[idx]
            text = text_by_index.get(idx, "")
            label = labels.get(qid, "")
            desc = descriptions.get(qid, "")
            rows.append((idx, qid, label, desc, text))
        if args.print_all:
            for idx, qid, label, desc, text in rows:
                print(f"[{idx}] {qid}\tLabel: {label}\tDesc: {desc}\tText: {text}")
        else:
            k = max(0, int(args.max_print))
            print(f"Showing first {k} example nodes:")
            for idx, qid, label, desc, text in rows[:k]:
                print(f"[{idx}] {qid}\tLabel: {label}\tDesc: {desc}\tText: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


