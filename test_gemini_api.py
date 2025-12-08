#!/usr/bin/env python3
"""
Minimal Gemini API smoke test.

Usage examples:
  GEMINI_API_KEY=... python3 test_gemini_api.py --prompt "Hello Gemini"
  python3 test_gemini_api.py --prompt-file prompt.txt --model gemini-1.5-flash
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import re
from typing import Optional


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Test Google Gemini API")
    ap.add_argument("--prompt", default="", help="Inline prompt text")
    ap.add_argument("--prompt-file", default="", help="Read prompt from file path")
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"), help="Gemini model name")
    ap.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY", ""), help="Gemini API key (or set GEMINI_API_KEY)")
    ap.add_argument("--max-output-tokens", type=int, default=512, help="Max output tokens")
    ap.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    ap.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling")
    ap.add_argument("--retries", type=int, default=3, help="Retries on 429/5xx with backoff")
    return ap.parse_args()


def _load_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    # Fallback: read from stdin if available
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return "Say hello and report the current year."


def _extract_retry_seconds(error_text: str) -> Optional[float]:
    # Try to extract "retry in Xs" from error text
    m = re.search(r"retry\s+in\s+([0-9]+(?:\.[0-9]+)?)s", error_text, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    # Try protobuf-like field "retry_delay { seconds: 43 }"
    m2 = re.search(r"retry_delay\s*\{\s*seconds:\s*([0-9]+)\s*\}", error_text, flags=re.IGNORECASE)
    if m2:
        try:
            return float(m2.group(1))
        except Exception:
            return None
    return None


def call_gemini_with_retry(
    model_name: str,
    api_key: str,
    prompt: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    retries: int,
) -> str:
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        return "google-generativeai not installed. Install with: pip install google-generativeai"

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    # Generation config is optional; set common fields
    gen_cfg = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    attempt = 0
    backoff = 10.0
    while True:
        attempt += 1
        try:
            resp = model.generate_content(
                prompt,
                generation_config=gen_cfg,  # type: ignore[arg-type]
            )
            try:
                txt = resp.text
            except Exception:
                txt = str(resp)
            return txt if txt else "(no text in response)"
        except Exception as e:
            etxt = str(e)
            # Detect rate-limit 429 or transient 5xx; otherwise return immediately
            transient = "429" in etxt or "ResourceExhausted" in etxt or "rate" in etxt.lower() or "quota" in etxt.lower() or "500" in etxt or "503" in etxt
            if not transient or attempt > max(0, int(retries)):
                return f"Gemini error (attempt {attempt}): {e}"
            # Respect retry hint if present; otherwise exponential backoff
            hinted = _extract_retry_seconds(etxt)
            wait_s = hinted if hinted is not None else backoff
            wait_s = max(1.0, min(wait_s, 120.0))
            time.sleep(wait_s)
            backoff = min(backoff * 2.0, 120.0)


def main() -> int:
    args = parse_args()
    prompt = _load_prompt(args).strip()
    if not prompt:
        print("Empty prompt. Provide --prompt or --prompt-file, or pipe input.", file=sys.stderr)
        return 2
    api_key = args.api_key.strip()
    if not api_key:
        print("Gemini API key not provided. Set --api-key or GEMINI_API_KEY env var.", file=sys.stderr)
        return 2
    out = call_gemini_with_retry(
        model_name=args.model,
        api_key=api_key,
        prompt=prompt,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        retries=args.retries,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


