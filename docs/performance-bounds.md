# HY4 Flash performance bounds

The Spark Fast GGUF header contains 83.576 GB of routed experts, 4.126 GB of other backbone tensors, 0.209 GB for the LM head, 0.394 GB for the token table, 67.321 GB for the disk-backed Engram table, and 3.980 GB in draft blocks. These figures derive from file offsets and include alignment. The router's top-16 out of 512 suggests about 2.612 GB of expert weights read per isolated token. Counting the backbone and head as one read gives a total of about 6.947 GB per non-speculative step at short context, before token/Engram row lookups and attention-cache traffic. The 273 GB/s hardware specification implies about 39.3 tok/s under that model. This is arithmetic, not a measured throughput or an MTP ceiling.

The device probe reads and writes two large buffers on CUDA, then times a large FP16 GEMM. The roofline script substitutes measured copy throughput, bounded by NVIDIA's memory rating, and reports an illustrative prefill estimate at 50% of measured GEMM efficiency. Sparse attention, quantized kernels, the indexer, and the CPU/SSD lookup complicate that estimate. Use the measured baseline and identical control prompts to decide whether a change works. [NVIDIA hardware guide](https://docs.nvidia.com/dgx/dgx-spark/hardware.html) · [Tencent architecture](https://github.com/Tencent-Hunyuan/Hy4-flash#model-introduction) · [Inference-optimisation methodology](https://github.com/sybil-solutions/inference-optimisation-skill/blob/fda968e7fc1996aa23c67139c755fe541c9d4be0/skills/inference-optimisation/references/ceilings.md)

## Reproduce the probes

Run the CUDA probe on the Spark while the model server is stopped. It allocates just over 1 GiB of device buffers. Pass the verified checkpoint path to the header reader.

```bash
"${CUDACXX:-/usr/local/cuda-13.0/bin/nvcc}" -arch=sm_121a -O3 bin/probe.cu -lcublas -o /tmp/hy4-probe
/tmp/hy4-probe > device.json
python3 -B bin/weights.py "${HY4_DATA_DIR:-$HOME/.cache/hy4-flash-spark}/Hy4-flash-SPARK-4Bit-Fast.gguf" > weights.json
python3 -B bin/roofline.py device.json weights.json
```
