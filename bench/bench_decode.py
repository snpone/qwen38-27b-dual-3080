#!/usr/bin/env python3
"""
Qwen3.8-27B decode speed benchmark — pure decode tok/s (excludes prefill/TTFT).

Usage:
    python bench_decode.py [--base http://127.0.0.1:8000] [--model qwen3.8-27b]
                          [--max-tokens 400] [--rounds 6]

Measures:
- Wall time, TTFT, completion tokens
- Decode rate = completion / (wall - TTFT)  ← the key metric
- Reports median + range across rounds (min 6 rounds recommended)

Output: JSON to stdout
"""
import argparse, json, os, sys, time, urllib.request, statistics

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

PROMPTS = {
    "code": "Write a Python function that implements a thread-safe LRU cache with TTL eviction. Include type hints and docstrings. Use only stdlib.",
    "chinese": "用中文详细写一篇关于量子计算的科普文章，内容要详实、多段落，涵盖量子比特、叠加态、纠缠和量子计算优势。",
    "count": "Count from 1 to 1000, one number per line. Start with 1.",
}


def stream_chat(base, model, prompt, max_tokens=400, timeout=600):
    """Stream a chat completion. Returns (wall_s, ttft_s, completion_tokens)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    ttft = None
    n_out = 0
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
                n_out += 1
            if d.get("usage"):
                usage = d["usage"]
    wall = time.time() - t0
    ct = usage.get("completion_tokens", n_out) if usage else n_out
    return wall, ttft, ct


def bench_one(base, model, task, max_tokens, timeout):
    prompt = PROMPTS[task]
    wall, ttft, ct = stream_chat(base, model, prompt, max_tokens, timeout)
    decode_t = (wall - ttft) if ttft else wall
    decode = round(ct / max(decode_t, 0.001), 1)
    return {
        "task": task,
        "wall_s": round(wall, 2),
        "ttft_s": round(ttft, 3) if ttft else None,
        "completion_tokens": ct,
        "decode_tok_s": decode,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="qwen3.8-27b")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--tasks", nargs="+", default=list(PROMPTS.keys()))
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()

    # Warm up
    print(f"Warming up ({args.model})...", file=sys.stderr)
    stream_chat(args.base, args.model, "Hi", 50, 120)
    time.sleep(1)

    results = {"config": {"model": args.model, "max_tokens": args.max_tokens,
                           "rounds": args.rounds, "tasks": args.tasks},
               "per_round": [], "summary": {}}

    for task in args.tasks:
        task_rates = []
        for i in range(args.rounds):
            r = bench_one(args.base, args.model, task, args.max_tokens, args.timeout)
            results["per_round"].append(r)
            task_rates.append(r["decode_tok_s"])
            print(f"  [{task}] round {i+1}/{args.rounds}: "
                  f"decode={r['decode_tok_s']} tok/s  "
                  f"TTFT={r['ttft_s']}s  wall={r['wall_s']}s  n={r['completion_tokens']}",
                  file=sys.stderr, flush=True)
            time.sleep(1)
        med = statistics.median(task_rates)
        results["summary"][task] = {
            "median_tok_s": med,
            "mean_tok_s": round(statistics.mean(task_rates), 1),
            "min_tok_s": min(task_rates),
            "max_tok_s": max(task_rates),
            "n_rounds": len(task_rates),
        }

    # Print summary
    print("\n=== Summary ===")
    for task, s in results["summary"].items():
        print(f"  {task:>10}: median={s['median_tok_s']:>6.1f}  "
              f"mean={s['mean_tok_s']:>6.1f}  "
              f"range={s['min_tok_s']}-{s['max_tok_s']}  "
              f"n={s['n_rounds']}")
    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
