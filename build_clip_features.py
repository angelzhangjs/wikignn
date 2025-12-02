#!/usr/bin/env python3
"""
Build CLIP-based node and edge embeddings from a saved .pt graph (export_to_pt.py).

Outputs a new .pt with:
- pyg_data (HeteroData) populated with:
  - data[ntype].x: CLIP embeddings for nodes of type ntype
  - data[(src, rel, dst)].edge_attr: CLIP embeddings for edges (per-relation text, repeated per edge)
Also preserves original payload keys.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Iterable

import torch

try:
    from torch_geometric.data import HeteroData  # type: ignore
    HAS_PYG = True
except Exception:
    HAS_PYG = False

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Add CLIP embeddings to saved .pt graph")
    ap.add_argument("--pt_in", required=True, help="Input .pt produced by export_to_pt.py")
    ap.add_argument("--out", required=True, help="Output .pt path to save with CLIP features")
    ap.add_argument("--graph_dir", default="", help="Graph dir to fetch optional labels (nodes.jsonl).")
    ap.add_argument("--backend", choices=["open_clip", "clip"], default="open_clip", help="CLIP backend to use")
    ap.add_argument("--model", default="ViT-B-32", help="CLIP model (open_clip: ViT-B-32; clip: ViT-B/32)")
    ap.add_argument("--pretrained", default="openai", help="Pretrained tag (open_clip only, e.g., openai or laion2b_s34b_b79k)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run encoding")
    ap.add_argument("--batch_size", type=int, default=256, help="Batch size for text encoding")
    ap.add_argument("--fp16", action="store_true", help="Encode in float16 for speed/memory")
    ap.add_argument("--prop_lang", default="en", help="Language code for property labels (e.g., en, de)")
    ap.add_argument("--prop_labels_path", default="", help="Optional JSON file mapping property_id -> label")
    ap.add_argument("--no_fetch_labels", action="store_true", help="Do not fetch property labels from Wikidata")
    return ap.parse_args()

def iter_jsonl(path: str) -> Iterable[dict]:
    if not path or not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield torch.serialization.json.loads(line)  # type: ignore[attr-defined]
            except Exception:
                # Fallback to stdlib if torch's JSON is unavailable
                import json as _json  # local import
                try:
                    yield _json.loads(line)
                except Exception:
                    continue

def load_optional_labels(graph_dir: str) -> Dict[str, str]:
    """
    Try to load human-readable labels for entity nodes from nodes.jsonl.
    Looks for fields commonly named 'label' or 'name'. Falls back to qid.
    """
    labels: Dict[str, str] = {}
    nodes_path = os.path.join(graph_dir, "nodes.jsonl")
    for row in iter_jsonl(nodes_path):
        qid = row.get("qid")
        if not isinstance(qid, str):
            continue
        label = (
            row.get("label")
            or row.get("name")
            or (isinstance(row.get("labels"), dict) and (row.get("labels") or {}).get("en"))
        )
        if isinstance(label, str) and label.strip():
            labels[qid] = label.strip()
        else:
            labels[qid] = qid
    return labels


def ensure_pyg(payload: Dict[str, object]) -> HeteroData:
    if "pyg_data" in payload and isinstance(payload["pyg_data"], HeteroData):  # type: ignore[arg-type]
        return payload["pyg_data"]  # type: ignore[return-value]
    if not HAS_PYG:
        raise RuntimeError("torch_geometric not available; cannot build HeteroData. Install PyG or run export with --format pyg.")
    node_index: Dict[str, Dict[str, int]] = payload["node_index"]  # type: ignore[assignment]
    edge_index: Dict[str, torch.Tensor] = payload["edge_index"]  # type: ignore[assignment]
    data = HeteroData()
    for ntype, idx_map in node_index.items():
        data[ntype].num_nodes = len(idx_map)
    for key_str, ei in edge_index.items():
        src_t, rel, dst_t = key_str.split(":")
        data[(src_t, rel, dst_t)].edge_index = ei
    payload["pyg_data"] = data
    return data


class ClipEncoder:
    def __init__(self, backend: str, model: str, pretrained: str, device: str, fp16: bool) -> None:
        self.backend = backend
        self.device = device
        self.fp16 = fp16
        self.dtype = torch.float16 if fp16 else torch.float32

        self._open_clip = None
        self._clip = None
        self.model = None
        self.tokenize = None

        if backend == "open_clip":
            try:
                import open_clip  # type: ignore
            except Exception as e:
                raise RuntimeError("open_clip is not installed. Try: pip install open_clip_torch") from e
            self._open_clip = open_clip
            self.model, _, _ = open_clip.create_model_and_transforms(model, pretrained=pretrained, device=device)
            self.tokenize = open_clip.get_tokenizer(model)
        else:
            try:
                import clip  # type: ignore
            except Exception as e:
                raise RuntimeError("clip is not installed. Try: pip install git+https://github.com/openai/CLIP.git") from e
            self._clip = clip
            # Map common name if user passed ViT-B-32
            model_name = model if "/" in model else model.replace("-", "/")
            self.model, _ = clip.load(model_name, device=device)
            self.tokenize = clip.tokenize

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def encode_texts(self, texts: List[str], batch_size: int) -> torch.Tensor:
        if not texts:
            return torch.empty((0, 0), dtype=self.dtype, device="cpu")
        all_embs: List[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokens = self.tokenize(batch).to(self.device)  # type: ignore[operator]
            if self.backend == "open_clip":
                text_features = self.model.encode_text(tokens)  # type: ignore[attr-defined]
            else:
                text_features = self.model.encode_text(tokens)  # type: ignore[union-attr]
            text_features = text_features.to(dtype=self.dtype)
            # Normalize for cosine similarity friendliness
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
            all_embs.append(text_features)
        embs = torch.cat(all_embs, dim=0).to("cpu", dtype=torch.float32)
        return embs

def build_node_texts(node_index: Dict[str, Dict[str, int]], entity_labels: Dict[str, str]) -> Dict[str, List[str]]:
    texts_by_type: Dict[str, List[str]] = {}
    # entity: use label if available, else qid
    ent_map = node_index.get("entity", {})
    rev_entity = {idx: qid for qid, idx in ent_map.items()}
    ent_texts = []
    for idx in range(len(ent_map)):
        qid = rev_entity.get(idx, f"Q{idx}")
        label = entity_labels.get(qid, qid)
        ent_texts.append(label)
    texts_by_type["entity"] = ent_texts

    # literal: parse "lit:{prop}:{value}"
    lit_map = node_index.get("literal", {})
    rev_lit = {idx: lid for lid, idx in lit_map.items()}
    lit_texts = []
    for idx in range(len(lit_map)):
        lid = rev_lit.get(idx, f"lit:{idx}")
        parts = lid.split(":", 2)
        if len(parts) == 3:
            _, prop, value = parts
            lit_texts.append(f"{value} ({prop})")
        else:
            lit_texts.append(lid)
    texts_by_type["literal"] = lit_texts

    # external_id: parse "ext:{prop}:{value}"
    ext_map = node_index.get("external_id", {})
    rev_ext = {idx: eid for eid, idx in ext_map.items()}
    ext_texts = []
    for idx in range(len(ext_map)):
        eid = rev_ext.get(idx, f"ext:{idx}")
        parts = eid.split(":", 2)
        if len(parts) == 3:
            _, prop, value = parts
            ext_texts.append(f"{prop} {value}")
        else:
            ext_texts.append(eid)
    texts_by_type["external_id"] = ext_texts
    return texts_by_type

def build_relation_text(key_str: str) -> str:
    src_t, prop, dst_t = key_str.split(":")
    return f"{prop} relation from {src_t} to {dst_t}"

def build_relation_text_with_labels(key_str: str, pid_to_label: Dict[str, str]) -> str:
    src_t, prop, dst_t = key_str.split(":")
    label = pid_to_label.get(prop, prop)
    return f"{label} relation from {src_t} to {dst_t}"

def collect_property_ids(edge_index: Dict[str, torch.Tensor]) -> List[str]:
    pids: List[str] = []
    seen: set = set()
    for key_str in edge_index.keys():
        _, pid, _ = key_str.split(":")
        if pid not in seen:
            seen.add(pid)
            pids.append(pid)
    return pids

def load_property_labels_json(path: str) -> Dict[str, str]:
    if not path:
        return {}
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}

def fetch_property_labels(pids: List[str], lang: str) -> Dict[str, str]:
    try:
        import requests  # type: ignore
    except Exception:
        return {}
    labels: Dict[str, str] = {}
    for i in range(0, len(pids), 50):
        chunk = pids[i:i+50]
        try:
            r = requests.get(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbgetentities",
                    "ids": "|".join(chunk),
                    "props": "labels",
                    "languages": lang,
                    "format": "json",
                },
                timeout=20,
            )
            j = r.json()
            ents = j.get("entities", {}) if isinstance(j, dict) else {}
            for pid, ent in ents.items():
                if not isinstance(ent, dict):
                    continue
                val = ent.get("labels", {}).get(lang, {}).get("value")
                if isinstance(val, str) and val.strip():
                    labels[pid] = val.strip()
        except Exception:
            continue
    return labels

def main() -> int:
    args = parse_args()

    payload = torch.load(args.pt_in, map_location="cpu")
    if not isinstance(payload, dict):
        print("Input .pt is not a dict payload.", file=sys.stderr)
        return 2

    if not HAS_PYG:
        print("torch_geometric not available; will try to proceed only if pyg_data already exists.", file=sys.stderr)
    data = ensure_pyg(payload)

    node_index: Dict[str, Dict[str, int]] = payload["node_index"]  # type: ignore[assignment]
    edge_index: Dict[str, torch.Tensor] = payload["edge_index"]  # type: ignore[assignment]

    # Optional labels
    entity_labels: Dict[str, str] = load_optional_labels(args.graph_dir) if args.graph_dir else {}

    # CLIP encoder
    encoder = ClipEncoder(
        backend=args.backend,
        model=args.model,
        pretrained=args.pretrained,
        device=args.device,
        fp16=args.fp16,
    )

    # 1) Node embeddings
    texts_by_type = build_node_texts(node_index, entity_labels)
    for ntype, texts in texts_by_type.items():
        if not texts:
            continue
        x = encoder.encode_texts(texts, batch_size=args.batch_size)
        data[ntype].x = x  # [num_nodes, d]
        print(f"Encoded nodes for type '{ntype}': {x.shape}")

    # 2) Edge embeddings (per relation type, repeated per edge)
    pid_to_label: Dict[str, str] = {}
    if args.prop_labels_path:
        pid_to_label.update(load_property_labels_json(args.prop_labels_path))
    if not args.no_fetch_labels:
        missing = [pid for pid in collect_property_ids(edge_index) if pid not in pid_to_label]
        if missing:
            fetched = fetch_property_labels(missing, args.prop_lang)
            pid_to_label.update(fetched)
    for key_str, ei in edge_index.items():
        rel_text = build_relation_text_with_labels(key_str, pid_to_label) if pid_to_label else build_relation_text(key_str)
        rel_vec = encoder.encode_texts([rel_text], batch_size=1)  # [1, d]
        src_t, rel, dst_t = key_str.split(":")
        E = int(ei.size(1))
        edge_attr = rel_vec.repeat(E, 1).contiguous()  # [E, d]
        data[(src_t, rel, dst_t)].edge_attr = edge_attr
        print(f"Encoded edges for relation '{key_str}': edge_attr {edge_attr.shape}")

    # Save updated payload
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload["pyg_data"] = data
    torch.save(payload, args.out)
    print(f"Saved CLIP-augmented graph to: {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


