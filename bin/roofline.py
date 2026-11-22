#!/usr/bin/env python3
"""Estimate HY4's short-context decode bandwidth bound from the hardware and GGUF probes."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device", type=Path)
    parser.add_argument("weights", type=Path)
    args = parser.parse_args()
    device = json.loads(args.device.read_text())
    file = json.loads(args.weights.read_text())
    # The probe counts both source reads and destination writes.
    bandwidth = min(273.0, device["cuda_copy_read_plus_write_GB_s"])
    decode_bytes = file["active_routed_weight_byte_estimate"] + file["backbone_and_head_byte_estimate"]
    # The two weighted value products approximate attention arithmetic. Omit indexer
    # selection and Engram access, which do not behave like one large GEMM.
    max_attended = 30 * 64 * 256 * 512 + 12 * 32 * 256 * 2048
    prefill_flops = 2 * 11.3e9 + 4 * max_attended
    print(json.dumps({
        "bandwidth_proxy_GB_s": bandwidth,
        "non_speculative_short_context_read_bytes_per_token_estimate": decode_bytes,
        "bandwidth_model_tok_s": bandwidth * 1e9 / decode_bytes,
        "illustrative_prefill_tok_s_at_50_percent_fp16_gemm_efficiency":
            0.5 * device["fp16_gemm_TFLOP_s"] * 1e12 / prefill_flops,
        "scope": "Diagnostic estimates only. Ignores Engram reads, indexer work, kernel overhead, "
                 "cache traffic and MTP. GEMM measures FP16 while the actual weights are mixed precision.",
    }, indent=2))


if __name__ == "__main__":
    main()
