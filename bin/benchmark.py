#!/usr/bin/env python3
"""Direct loopback timing and retrieval for the HY4 Flash llama.cpp recipe."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
from pathlib import Path
import random
import re
import statistics
import sys
import threading
import time

HOST = "127.0.0.1"
MODEL = "hy4-flash"
PROSE = ("Write a detailed step-by-step explanation of how a hash map works, "
         "including collision handling, resizing, and time complexity. Be thorough.")
REGIONS = ("cedar", "bay", "canyon", "delta", "field", "grove", "hill", "lake")


def request(port, path, payload=None, timeout=60):
    with_run = http.client.HTTPConnection(HOST, port, timeout=timeout)
    try:
        method, body = ("GET", None) if payload is None else ("POST", json.dumps(payload).encode())
        with_run.request(method, path, body, {"Content-Type": "application/json"})
        response = with_run.getresponse()
        if response.status != 200:
            raise RuntimeError(f"{path} HTTP {response.status}: {response.read(480).decode(errors='replace')}")
        data = response.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            raise RuntimeError(f"{path} gave an oversized response")
        return data
    finally:
        with_run.close()


def reasoning_options(thinking):
    return {"reasoning_effort": "high" if thinking else "no_think",
            "chat_template_kwargs": {"enable_thinking": thinking}}


def token_count(port, prompt):
    chat = {"messages": [{"role": "user", "content": prompt}], **reasoning_options(False)}
    rendered = json.loads(request(port, "/apply-template", chat))["prompt"]
    return len(json.loads(request(port, "/tokenize", {
        "content": rendered, "add_special": True, "parse_special": True,
    }))["tokens"])


def stream(port, prompt, limit, thinking, force_length=False, barrier=None):
    payload = {
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": limit, "stream": True, "stream_options": {"include_usage": True},
        "temperature": 0.7 if thinking else 0, "top_k": 20, "top_p": 1,
        **reasoning_options(thinking),
        "cache_prompt": thinking, "ignore_eos": force_length,
    }
    con = http.client.HTTPConnection(HOST, port, timeout=720)
    timings, usage, text = {}, {}, []
    finish = None
    first = last = None
    saw_reasoning = False
    if barrier is not None:
        barrier.wait(timeout=30)
    start = time.monotonic()
    try:
        con.request("POST", "/v1/chat/completions", json.dumps(payload).encode(),
                    {"Content-Type": "application/json", "Accept": "text/event-stream"})
        response = con.getresponse()
        if response.status != 200:
            raise RuntimeError(f"Chat HTTP {response.status}: {response.read(480).decode(errors='replace')}")
        while True:
            if time.monotonic() - start > 720:
                raise RuntimeError("Chat exceeded 720 seconds")
            line = response.readline(256 * 1024 + 1)
            if not line or len(line) > 256 * 1024:
                raise RuntimeError("Unexpected end or size of SSE data")
            if not line.startswith(b"data:"):
                continue
            chunk = line[5:].strip()
            if chunk == b"[DONE]":
                break
            entry = json.loads(chunk)
            if "error" in entry:
                raise RuntimeError(f"Chat stream failed: {entry['error']!r}")
            for choice in entry.get("choices") or []:
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or {}
                content = delta.get("content")
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if content or reasoning:
                    last = time.monotonic()
                    first = first if first is not None else last
                    saw_reasoning |= bool(reasoning)
                if content:
                    text.append(content)
            usage.update(entry.get("usage") or {})
            timings.update(entry.get("timings") or {})
    finally:
        con.close()
    n_generated = usage.get("completion_tokens", timings.get("predicted_n"))
    prompt_tokens = usage.get("prompt_tokens")
    if first is None or finish not in {"stop", "length"} or not n_generated or not prompt_tokens:
        raise RuntimeError("Missing stream text, terminal status, or API token counts")
    if force_length and (n_generated != limit or not saw_reasoning):
        raise RuntimeError(f"Prose did not return {limit} tokens with thinking on")
    start_rate = (n_generated - 1) / (last - first) if last > first else None
    return {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "completion_tokens": n_generated, "prompt_tokens": prompt_tokens,
            "ttft_s": first - start, "decode_tok_s": start_rate, "finish_reason": finish,
            "reasoning_seen": saw_reasoning, "timings": timings,
            "first": first, "last": last}, "".join(text)


def run_decode(port, users, repeats):
    stream(port, PROSE, 32, True)
    waves = []
    for index in range(repeats):
        prompts = [PROSE] if users == 1 else [f"Request ID: decode-{n}\n{PROSE}" for n in range(1, users + 1)]
        barrier = threading.Barrier(users)
        with ThreadPoolExecutor(max_workers=users) as pool:
            requests = [pool.submit(stream, port, prompt, 400, True, True, barrier) for prompt in prompts]
            rows = [job.result()[0] for job in requests]
        aggregate = sum(row["completion_tokens"] - 1 for row in rows) / (
            max(row["last"] for row in rows) - min(row["first"] for row in rows)
        )
        overlap = min(row["last"] for row in rows) - max(row["first"] for row in rows)
        first_arrival = min(row["first"] for row in rows)
        for row in rows:
            row["first_offset_s"] = row.pop("first") - first_arrival
            row["last_offset_s"] = row.pop("last") - first_arrival
        if users > 1 and overlap <= 0:
            raise RuntimeError("The requested streams did not all decode concurrently")
        waves.append({"wave": index + 1, "aggregate_decode_tok_s": aggregate,
                      "all_users_overlap_s": overlap, "requests": rows})
        print(f"Prose C{users} {index + 1}/{repeats} {aggregate:.2f} tok/s", flush=True)
    return {"users": users, "temperature": 0.7, "top_k": 20, "top_p": 1, "thinking": True, "reasoning_effort": "high",
            "median_decode_tok_s": statistics.median(row["aggregate_decode_tok_s"] for row in waves),
            "waves": waves}


def long_prompt(seed, rows):
    rng = random.Random(seed)
    batch = rng.getrandbits(64)
    marker = f"FROST-{rng.getrandbits(32):08X}"
    records = [f"Batch {batch:016x}. These records are synthetic.",
               "Find the audit code and reply with the code only."]
    for n in range(rows):
        if n == rows // 2:
            records.append(f"AUDIT CODE: {marker}.")
        records.append(f"record {n:05d} | hub {rng.choice(REGIONS)} "
                       f"| units {rng.randrange(100,1000)} | lot {rng.getrandbits(32):08x}")
    records.append("What is the audit code? Reply with the code only.")
    return "\n".join(records), marker


def fresh_prefill(port, target, seed):
    probe, _ = long_prompt(seed, 64)
    samples = round(target * 64 / token_count(port, probe))
    for _ in range(4):
        prompt, marker = long_prompt(seed, samples)
        count = token_count(port, prompt)
        if abs(count - target) <= target * 0.005:
            break
        samples = round(samples * target / count)
    else:
        raise RuntimeError(f"Could not size a {target}-token input")
    print(f"Cold prefill at {count:,} input tokens", flush=True)
    row, text = stream(port, prompt, 32, False)
    timing = row["timings"]
    if abs(row["prompt_tokens"] - target) > target * 0.015 or timing.get("cache_n") != 0:
        raise RuntimeError("Prefill run lacked the target prompt size or zero cache hits")
    row.update(target_prompt_tokens=target, sized_prompt_tokens=count,
               audit_code_matched=text.strip() == marker,
               engine_prefill_tok_s=1000 * timing["prompt_n"] / timing["prompt_ms"],
               client_prefill_tok_s=row["prompt_tokens"] / row["ttft_s"])
    del row["first"], row["last"]
    return row


def metrics(port):
    names = ("spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total",
             "spec_decode_num_drafts_total")
    lines = request(port, "/metrics").decode().splitlines()
    return {name: sum(float(line.rsplit(None, 1)[1]) for line in lines
                      if re.match(r"(?:llamacpp:|llamacpp_)" + name + r"(?:\{|\s)", line))
            for name in names}


def write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18911)
    parser.add_argument("--users", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--scenario", choices=("decode", "prefill"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert 1024 <= args.port <= 65535 and args.repeats > 0
    assert not args.output.exists(), args.output
    models = json.loads(request(args.port, "/v1/models"))
    assert MODEL in {row["id"] for row in models["data"]}, models
    result = {"engine": "llama.cpp", "endpoint": f"http://{HOST}:{args.port}",
              "model": MODEL, "scenario": args.scenario}
    if args.scenario == "decode":
        before = metrics(args.port)
        result.update(run_decode(args.port, args.users, args.repeats))
        after = metrics(args.port)
        result["spec_counters_delta_including_warmup"] = {key: after[key] - before[key] for key in before}
    else:
        result["prefill"] = []
        for size in (32768, 131072):
            result["prefill"].append(fresh_prefill(args.port, size, 20261122 + size))
            write_report(args.output, result)
    write_report(args.output, result)


if __name__ == "__main__":
    main()
