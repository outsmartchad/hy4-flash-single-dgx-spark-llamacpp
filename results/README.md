# Raw results

The JSON files retain each request's prompt hash, token counts and API timing fields. The memory CSVs contain Spark-local timestamps, Linux `MemAvailable`, process anonymous RSS and the NVIDIA driver's GPU allocation for the llama-server process.

- `control-a`, `control-b`, `control-c` are separate MTP ×3 starts. `mtp-off` and `mtp-five` are the interleaved variants.
- `configuration-selection.json` records the tested selection rule and its result.
- `final-decode.json` repeats the selected profile. `final-prefill.json` contains the two cold prompt runs and their exact retrieval checks.
- `c5-mtp3.json`, `c5-off.json` and `c5-mtp1.json` cover five concurrent short-prompt streams in three waves per profile. Request offsets and the all-user overlap verify simultaneous decode. These memory CSVs sample host headroom and the GPU process allocation.
- `checks.json` records the localhost chat, reasoning, Traditional Chinese and synthetic tool round trip on the first MTP ×3 server.
- `device.json` contains the CUDA copy and cuBLAS results. `weights.json` totals byte ranges from the verified GGUF tensor descriptors.
- [`tuning-20261123/`](tuning-20261123/) contains the startup, MTP depth, CUDA graph and qualification data for the follow-up. The `.startup-s` files record whole seconds from launch to loopback health. The memory CSVs also sample GPU utilization, temperature and graphics clocks.

[Method and thresholds](../docs/method.md) · [How to reproduce the startup and smoke checks](../docs/reproduce.md)
