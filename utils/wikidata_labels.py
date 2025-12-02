#!/usr/bin/env python3
"""
Utilities to fetch Wikidata property labels (language-specific).
"""
from __future__ import annotations

from typing import Dict, Iterable, List


def normalize_property_ids(pids: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for pid in pids:
        if not isinstance(pid, str):
            continue
        if not pid.startswith("P"):
            continue
        num = pid[1:]
        if not num.isdigit():
            continue
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def fetch_property_labels(pids: List[str], lang: str = "en", timeout: int = 20) -> Dict[str, str]:
    """
    Fetch human-readable labels for Wikidata property IDs in the given language.
    Returns mapping pid -> label (missing labels map to the pid itself).
    """
    try:
        import requests  # type: ignore
    except Exception as e:
        raise RuntimeError("The 'requests' package is required. Install with: pip install requests") from e

    import time

    labels: Dict[str, str] = {}
    pids = normalize_property_ids(pids)
    if not pids:
        return labels
    # Robust fetch parameters
    batch_size = 50
    max_retries = 4
    retry_backoff = 1.5
    headers = {
        "User-Agent": "angel-gnn/0.1 (+https://example.org; research; contact: user@example.org)"
    }
    for i in range(0, len(pids), batch_size):
        chunk = pids[i : i + batch_size]
        j = None
        for attempt in range(max_retries):
            try:
                r = requests.get(
                    "https://www.wikidata.org/w/api.php",
                    params={
                        "action": "wbgetentities",
                        "ids": "|".join(chunk),
                        "props": "labels",
                        "languages": lang,
                        "format": "json",
                        "formatversion": "2",
                    },
                    headers=headers,
                    timeout=timeout,
                )
                if r.status_code == 429:
                    ra = r.headers.get("Retry-After")
                    sleep_s = float(ra) if ra and ra.isdigit() else (retry_backoff * (2 ** attempt))
                    time.sleep(sleep_s)
                    continue
                r.raise_for_status()
                j = r.json()
                break
            except Exception:
                if attempt == max_retries - 1:
                    j = None
                else:
                    time.sleep(retry_backoff * (2 ** attempt))
        if not isinstance(j, dict):
            continue
        ents = j.get("entities", {}) if isinstance(j, dict) else {}
        for pid, ent in ents.items():
            if not isinstance(ent, dict):
                continue
            # formatversion=2: labels[lang] may be dict with 'value' or string depending on API
            lab = ent.get("labels", {}).get(lang)
            if isinstance(lab, dict):
                val = lab.get("value")
            else:
                val = lab
            labels[pid] = val if isinstance(val, str) and val.strip() else pid
    return labels


def load_json(path: str) -> Dict[str, str]:
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in (data.items() if isinstance(data, dict) else [])}


def save_json(obj: Dict[str, str], path: str) -> None:
    import json, os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



