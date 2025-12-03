#!/usr/bin/env python3
"""
Export entity metadata (labels/descriptions/aliases/text_by_index) to JSON files.

Examples:
  # Export from a labeled graph payload (.pt) that already contains entity_*_en maps
  python export_entity_metadata.py \
    --from_pt graph_output/clean_graph.pyg_en_labeled.en_labeled.pt \
    --lang en \
    --out_dir graph_output/entity_meta_en

  # Or export from a standalone metadata .pt (if you have one)
  python export_entity_metadata.py \
    --meta_pt path/to/entity_metadata_en.pt \
    --lang en \
    --out_dir graph_output/entity_meta_en
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any

import torch


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export entity metadata maps to JSON")
    ap.add_argument("--from_pt", default="", help="Graph payload .pt that may contain entity_*_{lang} maps")
    ap.add_argument("--meta_pt", default="", help="Standalone entity metadata .pt (optional)")
    ap.add_argument("--lang", default="en", help="Language code (default: en)")
    ap.add_argument("--out_dir", required=True, help="Output directory to write JSON files")
    return ap.parse_args()


def safe_get_maps(container: Dict[str, Any], lang: str) -> Dict[str, Any]:
    keys = [
        f"entity_labels_{lang}",
        f"entity_descriptions_{lang}",
        f"entity_aliases_{lang}",
        f"entity_text_{lang}_by_index",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        v = container.get(k)
        if isinstance(v, dict):
            out[k] = v
    return out


def write_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    merged: Dict[str, Any] = {}

    if args.from_pt:
        payload = torch.load(args.from_pt, map_location="cpu")
        if isinstance(payload, dict):
            merged.update(safe_get_maps(payload, args.lang))
    if args.meta_pt:
        meta = torch.load(args.meta_pt, map_location="cpu")
        if isinstance(meta, dict):
            merged.update(safe_get_maps(meta, args.lang))

    if not merged:
        raise SystemExit(f"No entity metadata found for lang='{args.lang}'. "
                         f"Check that your input .pt contains entity_*_{args.lang} maps.")

    # Write each present map to its own JSON
    for k, v in merged.items():
        out_path = os.path.join(args.out_dir, f"{k}.json")
        write_json(v, out_path)
        print(f"Wrote {k} -> {out_path} ({len(v)} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


