#!/usr/bin/env python3
"""
Prefix caching benchmark — measure TTFT reduction on repeated context.

Usage:
    python bench_prefix_cache.py [--base http://127.0.0.1:8000]
                                [--doc-size 25000]

Sends the same long document twice; second request should hit prefix cache.
"""
import argparse, json, os, sys, time, urllib.request

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

FILLER = (
    "In the fiscal year 2024, the Department of Transportation completed "
    "47 major infrastructure projects across 12 regions. Total expenditure "
    "was $2.3 billion, with a 15% reduction in average project delay. "
    "The quarterly reports highlighted improvements in procurement efficiency, "
    "workforce training outcomes, and vendor compliance rates. Each project "
    "underwent a three-stage review: feasibility, environmental impact, and "
    "public consultation. The final audit noted that 92% of projects were "
    "completed within budget, a 7-point improvement over the prior year. "
)


def build_doc(n_tokens):
    per = len(FILLER) // 4
    reps = max(1, n_tokens // per)
    return FILLER * reps


def stream_chat(base, model, prompt, max_tokens=100, timeout=600):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    ttft = None
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                d = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            c = d.get("choices")
            if c and c[0].get("delta", {}).get("content"):
                if ttft is None:
                    ttft = time.time() - t0
            if d.get("usage"):
                usage = d["usage"]
    wall = time.time() - t0
    return wall, ttft


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="qwen3.8-27b")
    p.add_argument("--doc-size", type=int, default=25000,
                    help="Target document size in tokens")
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()

    doc = build_doc(args.doc_size)
    print(f"Document size: ~{args.doc_size} tokens ({len(doc)} chars)", file=sys.stderr)

    # Round 1: cold
    wall1, ttft1 = stream_chat(args.base, args.model,
                                doc + "\n\nSummarize this in one sentence.",
                                100, args.timeout)
    if ttft1 is None:
        print("ERROR: TTFT not captured (request failed)", file=sys.stderr)
        sys.exit(1)
    print(f"Round 1 (cold):  TTFT={ttft1:.2f}s  wall={wall1:.2f}s", file=sys.stderr)

    time.sleep(2)

    # Round 2: should hit prefix cache
    wall2, ttft2 = stream_chat(args.base, args.model,
                                doc + "\n\nSummarize this in one sentence.",
                                100, args.timeout)
    if ttft2 is None:
        print("ERROR: TTFT not captured (request failed)", file=sys.stderr)
        sys.exit(1)
    print(f"Round 2 (warm):  TTFT={ttft2:.2f}s  wall={wall2:.2f}s", file=sys.stderr)

    speedup = ttft1 / ttft2 if ttft2 > 0 else float("inf")
    print(f"\nPrefix cache speedup: {speedup:.1f}x", file=sys.stderr)

    print(json.dumps({
        "doc_tokens_approx": args.doc_size,
        "cold_ttft_s": round(ttft1, 2),
        "warm_ttft_s": round(ttft2, 2),
        "speedup_x": round(speedup, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
