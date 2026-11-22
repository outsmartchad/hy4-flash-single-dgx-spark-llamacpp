# Troubleshooting

## CUDA or build fails

Run `./bin/recipe.sh doctor` and verify that `nvcc --version` from the configured `CUDACXX` path is available. The launcher builds native ARM64 code for GB10. A different llama.cpp revision can reject the publisher's patch or interpret the same GGUF differently. Check the source commit, patch checksum and CMake error before changing any flags.

## Weight download was interrupted

Run the same `fetch` command again with the same data directory and variant. The launcher resumes the partial file and checks its SHA-256 before moving it into place. Inspect available SSD capacity first. A checksum mismatch is a failed download and the server should use a verified copy.

## Startup stalls or the Spark runs short of memory

Allow a few minutes for the initial weight load. Check that other inference servers have released their GPU allocations. The full GGUF is larger than the Spark's physical memory. The launcher maps the Engram table on the CPU and places the main compute layers on the GPU. Review `nvidia-smi`, `free -h`, and the llama.cpp load log together. Keep the checkpoint on the local SSD. A smaller `HY4_CONTEXT` reduces attention-state allocations but also lowers the input limit. Retest the exact long prompt after any context change.

## Chat works but tool calls or reasoning look wrong

Check the final `finish_reason`, reasoning field, and tool-call fields from `/v1/chat/completions`. Requesting thinking can delay visible answer text. The engine's model-aware template and parser are part of the deployed recipe. Changing the template, sampler, draft setting or quantization can change both output content and timing. [Checks used by the recipe](../bin/checks.py)

## The API cannot be reached from your laptop

The server listens on Spark loopback. Verify it from the Spark shell with `curl -fsS http://127.0.0.1:18911/health`. For a client on your laptop, an existing SSH connection can forward a local port to the Spark's loopback endpoint. [Setup and checks](reproduce.md)
