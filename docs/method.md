# Benchmark method

Run one patched llama.cpp server at a time on one DGX Spark. The test client sends requests directly to `127.0.0.1:18911`. Use the same pinned Fast GGUF, one slot, a 196,608-token context, F16 KV cache, the same CUDA options and the same SSD placement in all profiles.

## Thinking-on prose decode

The prompt asks for a detailed explanation of hash maps, including collisions, resizing and time complexity. Each profile starts a new server and runs an unscored warmup. Six measured requests then use the checkpoint's `reasoning_effort=high`, temperature 0.7, top-p 1, top-k 20 and a 400-token output budget. The client requires a reasoning event and the full output count. It counts tokens from both reasoning and visible output. TTFT measures the time from sending the request until the first reasoning or text event. Client decode rate is 399 divided by the time between the first and last such events. Server token counts and engine timings are also in the raw output.

The MTP limit is the maximum number of draft proposals, with acceptance decided by the target. The metrics endpoint records draft tokens, accepted tokens and verification steps. The counters include the warmup request.

## Tuning order

This sequence describes the original November 22 tuning run. The [November 23 follow-up](tuning-2026-11-23.md) records the additional controls and the MTP ×2 result.

Use MTP ×3 as the repeated control and interleave the variants to measure drift between process restarts.

1. MTP ×3
2. MTP off
3. MTP ×3
4. MTP ×5
5. MTP ×3

Keep a candidate if its median decode is over 3% faster than each neighboring control and its median TTFT increases by at most 3% against each. Compare raw waves and check the selected setup separately. Keeping one checkpoint fixed isolates changes in speculative decoding within this runtime. The Spark Fast and Standard GGUFs have their own weight layouts. [Inference optimisation methodology](https://github.com/sybil-solutions/inference-optimisation-skill/tree/fda968e7fc1996aa23c67139c755fe541c9d4be0)

## Five concurrent users

The five-user follow-up measures short-prompt decode separately, with five slots and a total configured context of 196,608. A unique request identifier distinguishes each concurrent stream in the same wave. The client uses three waves per profile and timestamps each stream from first to last emitted token. The overlapping decode interval is the earliest last token minus the latest first token. The client rejects a wave if this interval is zero or negative. Aggregate decode uses total post-first tokens divided by the time from the earliest first token to the latest last token.

## Cold prefill and retrieval

For each of the 32K and 128K targets, generate a unique synthetic document containing a randomly seeded audit code in its middle. Ask the engine to apply its chat template, then size the document with its tokenizer. Send the document once with the checkpoint’s `reasoning_effort=no_think`, prefix-cache reuse disabled and a 32-token output limit. Require the actual prompt count to be within 1.5% of the target and the engine's cache count to be zero. “Cold” describes the prompt cache. It does not require dropping Linux page cache or reloading weights.

The benchmark checks the exact returned audit code. Engine prefill tok/s is processed input tokens divided by engine prefill time. Client prefill tok/s uses the total time to first token and includes request overhead, so the two can differ. Save the exact input count, TTFT, both rates, and retrieval result.

## Memory

Sample NVIDIA driver GPU process allocation and Linux `MemAvailable` approximately every two seconds. A sampled maximum may miss brief allocation spikes. Linux reports one shared physical memory pool. GPU allocation, file-backed host pages and estimated host headroom describe different parts of its use. Avoid adding GPU allocation to Linux process RSS.
