#!/usr/bin/env python3
"""Read GGUF tensor descriptors and estimate bytes touched by a short-context step."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import struct

WIDTHS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def gguf_weights(path):
    with path.open("rb") as file:
        def number(fmt):
            size = struct.calcsize("<" + fmt)
            return struct.unpack("<" + fmt, file.read(size))[0]

        def string():
            return file.read(number("Q")).decode("utf-8")

        def skip(kind):
            if kind == 8:
                file.seek(number("Q"), 1)
            elif kind == 9:
                value_type, count = number("I"), number("Q")
                if value_type in WIDTHS:
                    file.seek(WIDTHS[value_type] * count, 1)
                else:
                    for _ in range(count):
                        skip(value_type)
            else:
                file.seek(WIDTHS[kind], 1)

        if file.read(4) != b"GGUF" or number("I") != 3:
            raise ValueError("Expected a little-endian GGUF v3 file")
        tensors, fields = number("Q"), number("Q")
        alignment = 32
        for _ in range(fields):
            key, kind = string(), number("I")
            if key == "general.alignment" and kind == 4:
                alignment = number("I")
            else:
                skip(kind)
        offsets = []
        for _ in range(tensors):
            name, dimensions = string(), number("I")
            file.seek(8 * dimensions + 4, 1)
            offsets.append((number("Q"), name))
        begin = (file.tell() + alignment - 1) // alignment * alignment
    offsets.sort()
    stops = [x[0] for x in offsets[1:]] + [path.stat().st_size - begin]
    groups = defaultdict(int)
    for (start, name), stop in zip(offsets, stops):
        layer = re.match(r"blk\.(\d+)\.", name)
        if layer and int(layer[1]) >= 42:
            role = "draft"
        elif "_exps." in name:
            role = "routed_experts"
        elif name == "per_layer_token_embd.weight":
            role = "engram_table"
        elif name == "token_embd.weight":
            role = "token_embedding_table"
        elif name == "output.weight":
            role = "lm_head"
        else:
            role = "backbone_other"
        groups[role] += stop - start
    for required in ("engram_table", "routed_experts", "lm_head"):
        if groups[required] <= 0:
            raise ValueError(f"Missing expected HY4 weight group {required}")
    return {"file_bytes_by_group_with_alignment": dict(groups),
            "active_routed_weight_byte_estimate": groups["routed_experts"] // 32,
            "backbone_and_head_byte_estimate": groups["backbone_other"] + groups["lm_head"],
            "note": "The per-token estimate uses 16/512 expert selection, omits Engram and token embedding lookups, "
                    "and does not model attention-cache traffic, data reuse, or speculative decoding."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    args = parser.parse_args()
    print(json.dumps(gguf_weights(args.file), indent=2))


if __name__ == "__main__":
    main()
