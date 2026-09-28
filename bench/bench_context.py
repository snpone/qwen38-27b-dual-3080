#!/usr/bin/env python3
"""
Context gradient benchmark — decode rate vs prompt depth.

Tests how decode speed changes as context grows. Uses filler text
to build prompts of target sizes, then measures decode.

Usage:
    python bench_context.py [--base http://127.0.0.1:8000] [--model qwen3.8-27b]
                           [--ctxs 1000,10000,50000,100000,200000]

Output: JSON to stdout
"""
import argparse, json, os, sys, time, urllib.request, statistics, uuid

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

FILLER = (
    "The agency published a quarterly briefing on procurement delays, "
    "staffing gaps, and revised delivery dates for regional infrastructure projects. "
)
FILLER_PER_TOKEN = len(FILLER) // 4  # approx tokens per filler


def build_prompt(n_tokens):
    reps = max(1, n_tokens // FILLER_PER_TOKEN)
    return FILLER * reps + "\n[Task] Count from 1 to 30.\n"


def stream_chat(base, model, prompt, max_tokens=256, timeout=900):
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
    pt = usage.get("prompt_tokens", 0) if usage else 0
    ct = usage.get("completion_tokens", 0) if usage else 0
    decode_t = (wall - ttft) if ttft else wall
    decode = round(ct / max(decode_t, 0.001), 1)
    prefill = round(pt / max(ttft, 0.001), 1) if ttft else 0
    return {
        "target_ctx": n_tokens,
        "actual_prompt_tokens": pt,
        "wall_s": round(wall, 2),
        "ttft_s": round(ttft, 3) if ttft else None,
        "completion_tokens": ct,
        "decode_tok_s": decode,
        "prefill_tok_s": prefill,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="qwen3.8-27b")
    p.add_argument("--ctxs", default="1000,10000,50000,100000,200000")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--timeout", type=int, default=900)
    args = p.parse_args()

    ctxs = [int(x) for x in args.ctxs.split(",")]

    # Warm up
    print(f"Warming up ({args.model})...", file=sys.stderr)
    stream_chat(args.base, args.model, "Hi", 50, 120)
    time.sleep(1)

    results = {"config": {"model": args.model, "ctxs": ctxs, "rounds": args.rounds},
               "data": []}

    print("=== Context Gradient ===")
    for target in ctxs:
        best = None
        for i in range(args.rounds):
            prefix = f"[BENCH-{uuid.uuid4().hex[:8]}] "
            prompt = prefix + build_prompt(target)
            r = stream_chat(args.base, args.model, prompt, 256, args.timeout)
            results["data"].append(r)
            print(f"  ctx~{target:>7}: decode={r['decode_tok_s']:>6.1f}  "
                  f"prefill={r['prefill_tok_s']:>6.1f}  "
                  f"TTFT={r['ttft_s']:>8.2f}s  "
                  f"actual_pt={r['actual_prompt_tokens']}  "
                  f"(round {i+1}/{args.rounds})",
                  file=sys.stderr, flush=True)
            if best is None or r["decode_tok_s"] > best["decode_tok_s"]:
                best = r
            time.sleep(1)
        if best:
            print(f"  ctx~{target:>7}: BEST decode={best['decode_tok_s']:>6.1f} tok/s\n",
                  file=sys.stderr, flush=True)

    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
