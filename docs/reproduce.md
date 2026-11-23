# Build and run HY4 Flash on the Spark

Use one NVIDIA DGX Spark running ARM64 Linux. The build uses CMake, GCC, Git, curl, GNU coreutils, Python 3 and the CUDA 13.0 compiler. Run it on the Spark's terminal. The model file needs about 160 GB of local SSD space. Leave additional room for the source, build output, logs and temporary files.

```bash
./bin/recipe.sh doctor
./bin/recipe.sh build fast
./bin/recipe.sh fetch fast
./bin/recipe.sh serve fast
```

The launcher checks the model checksum after the download. An interrupted transfer leaves a `.part` file that the same fetch command can resume. The source checkout lives under the model data directory and is pinned to the upstream commit plus the publisher patch. Both the patch and the Hugging Face model file come from one pinned revision. The publisher provides the upstream model license. [AngelSlim release and Spark instructions](https://huggingface.co/AngelSlim/Hy4-flash-GGUF/tree/bcbc7db5dcf4162cc1e1b8ed6c65567e843ebe46) · [Tencent model license](https://huggingface.co/tencent/Hy4-flash/blob/b2498eba6a1a3f9bbcab4190bc1313a3c7010c2f/LICENSE)

Use `HY4_DATA_DIR` to put weights on a different local SSD. Point `CUDACXX` to the absolute compiler path if CUDA 13.0 lives somewhere other than `/usr/local/cuda-13.0/bin/nvcc`. The launcher compiles for `sm_121a`. The server listens on Spark loopback and uses the model ID `hy4-flash`. The measured profile took about three minutes to load. Once the health endpoint is ready, run these commands from another terminal on the Spark.

```bash
curl -fsS http://127.0.0.1:18911/health
curl -fsS http://127.0.0.1:18911/v1/models
curl -fsS http://127.0.0.1:18911/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hy4-flash","messages":[{"role":"user","content":"Explain what a hash table does in one sentence."}],"max_tokens":64,"reasoning_effort":"no_think","chat_template_kwargs":{"enable_thinking":false}}'
```

The launcher defaults to one slot and 196,608 configured context, with F16 KV, disk-backed Engram lookup and probabilistic MTP ×2. Set `LLAMA_ARG_SPEC_DRAFT_SAMPLING=greedy` to reproduce the previous one-user control. Add `HY4_DRAFT_MAX=3` to reproduce the original one-user baseline. `HY4_SPECULATION=off` disables speculation. The engine can clamp the draft length requested through `HY4_DRAFT_MAX`.

Set `HY4_LOAD_MODE=dio` to try Direct I/O for the upfront weight load. The Engram table still uses lazy memory mapping. The default is `mmap`. On Linux, the engine can fall back to buffered I/O when the filesystem rejects direct reads.

Run the quality checks and the two timing workloads against the same server:

```bash
python3 -B bin/checks.py --output checks.json
python3 -B bin/benchmark.py --scenario decode --users 1 --repeats 6 --output decode.json
python3 -B bin/benchmark.py --scenario prefill --output prefill.json
```

The tests use only synthetic prompts. The benchmark reports API token counts and the exact needle result separate from throughput. Prose decode includes thinking and counts tokens between the first and last emitted content or reasoning events. The prefill probe reports engine prefill and client TTFT for a unique prompt at each size.

## Five-user short-prompt decode

Run a separate server with five slots and the same total context budget:

```bash
HY4_PARALLEL=5 HY4_CONTEXT=196608 HY4_SPECULATION=off ./bin/recipe.sh serve fast
```

From a second terminal, send three waves of concurrent requests:

```bash
python3 -B bin/benchmark.py --scenario decode --users 5 --repeats 3 --output c5-off.json
```

For the MTP comparison, restart the server with `HY4_SPECULATION=mtp LLAMA_ARG_SPEC_DRAFT_SAMPLING=greedy` and `HY4_DRAFT_MAX=1` or `HY4_DRAFT_MAX=3`, then use a new output filename. The context budget is shared across slots. The 32K and 128K cold-prefill figures in the README were measured on the one-slot server.
