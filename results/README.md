# Raw results

The JSON files retain each request's prompt hash, token counts and API timing fields. The memory CSVs contain Spark-local timestamps, Linux `MemAvailable`, process anonymous RSS and the NVIDIA driver's GPU allocation for the llama-server process.

- `control-a`, `control-b`, `control-c` are separate MTP ×3 starts. `mtp-off` and `mtp-five` are the interleaved variants.
- `configuration-selection.json` records the tested selection rule and its result.
- `final-decode.json` repeats the selected profile. `final-prefill.json` contains the two cold prompt runs and their exact retrieval checks.
- `checks.json` records the localhost chat, reasoning, Traditional Chinese and synthetic tool round trip on the first MTP ×3 server.
- `device.json` contains the CUDA copy and cuBLAS results. `weights.json` totals byte ranges from the verified GGUF tensor descriptors.

[Method and thresholds](../docs/method.md) · [How to reproduce the startup and smoke checks](../docs/reproduce.md)
