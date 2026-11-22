#!/usr/bin/env python3
"""Small localhost chat, reasoning, language and tool round-trip checks."""
import argparse
import json
import re
from pathlib import Path

from benchmark import MODEL, reasoning_options, request, write_report

TOOL_NAME = "read_fixture_code"
TOOL_CODE = "H4X-LANTERN"
TOOLS = [{"type": "function", "function": {
    "name": TOOL_NAME, "description": "Read the synthetic fixture's audit code.",
    "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                   "required": ["name"], "additionalProperties": False},
}}]


def chat(port, messages, thinking=False, **extra):
    payload = {"model": MODEL, "messages": messages, "max_tokens": 512 if thinking else 96,
               "temperature": 0, **reasoning_options(thinking), **extra}
    data = json.loads(request(port, "/v1/chat/completions", payload, timeout=180))
    return data["choices"][0]


def user(text):
    return [{"role": "user", "content": text}]


def run_checks(port):
    result = {}
    first = chat(port, user("Reply with exactly HY4_OK and nothing else."))
    result["chat"] = {"finish_reason": first["finish_reason"],
                      "passed": (first["message"].get("content") or "").strip() == "HY4_OK"}
    reflection = chat(port, user("What is 37 + 48? Think first and answer with only the number."), True)
    msg = reflection["message"]
    answer = (msg.get("content") or "").strip()
    separate = bool(msg.get("reasoning_content") or msg.get("reasoning"))
    result["reasoning"] = {"separate_reasoning": separate, "passed": separate and answer == "85"}
    chinese = chat(port, user("請照抄這四個繁體中文字，除此之外不要輸出其他內容：學習愉快"))
    result["traditional_chinese"] = {"passed": (chinese["message"].get("content") or "").strip() == "學習愉快"}

    try:
        dialogue = user("Call read_fixture_code with the name lantern. Then reply with its audit code only.")
        chosen = chat(port, dialogue, tools=TOOLS,
                      tool_choice={"type": "function", "function": {"name": TOOL_NAME}})
        tool = chosen["message"]["tool_calls"][0]
        fields = json.loads(tool["function"]["arguments"])
        passed_call = tool["function"]["name"] == TOOL_NAME and fields.get("name") == "lantern"
        dialogue.append({"role": "assistant", "tool_calls": [tool], "content": ""})
        dialogue.append({"role": "tool", "tool_call_id": tool["id"], "name": TOOL_NAME,
                         "content": json.dumps({"audit_code": TOOL_CODE})})
        reply = chat(port, dialogue, tools=TOOLS)
        received = (reply["message"].get("content") or "").strip()
        result["tool_round_trip"] = {"called_expected_tool": passed_call,
                                     "passed": passed_call and bool(re.fullmatch(re.escape(TOOL_CODE), received))}
    except (KeyError, ValueError, IndexError, RuntimeError) as error:
        result["tool_round_trip"] = {"passed": False, "error": str(error)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18911)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run_checks(args.port)
    write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False))
    if not all(m["passed"] for m in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
