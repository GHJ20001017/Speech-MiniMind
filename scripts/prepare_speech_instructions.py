"""Build stage-2 speech instruction JSONL files from AISHELL-1 manifests.

AISHELL-1 provides reliable transcripts, so this script creates auditable
instruction examples without inventing facts that are absent from the audio.
The transcription task is the main task; text-operation tasks are small
auxiliary tasks for checking that the LLM follows instructions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TRANSCRIBE = (
    "请将这段语音准确转写为中文文本。",
    "听写这段语音，只输出你听到的中文内容。",
    "请把语音转换成文字，不要添加解释。",
)
TEXT_OPS = (
    ("请先转写这段语音，再告诉我这句话一共有多少个汉字。", "char_count"),
    ("请转写这段语音，并指出转写结果的第一个汉字。", "first_character"),
    ("请转写这段语音，并指出转写结果的最后一个汉字。", "last_character"),
)


def read_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def make_examples(rows: list[dict[str, str]], include_text_ops: bool) -> list[dict[str, str]]:
    examples = []
    for index, row in enumerate(rows):
        text = "".join(row["text"].split())
        prompt = TRANSCRIBE[index % len(TRANSCRIBE)]
        examples.append({
            "audio": row["path"],
            "instruction": prompt,
            "answer": text,
            "task": "transcription",
        })
        if include_text_ops and index % 5 == 0 and text:
            operation, task = TEXT_OPS[(index // 5) % len(TEXT_OPS)]
            if task == "char_count":
                answer = f"{text}\n这句话共有{len(text)}个汉字。"
            elif task == "first_character":
                answer = f"{text}\n第一个汉字是“{text[0]}”。"
            else:
                answer = f"{text}\n最后一个汉字是“{text[-1]}”。"
            examples.append({"audio": row["path"], "instruction": operation, "answer": answer, "task": task})
    return examples


def write_jsonl(path: Path, examples: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/aishell1/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/speech_instructions"))
    parser.add_argument("--include-text-ops", action="store_true", help="add auditable transcript-operation tasks")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"input": str(args.input), "output": str(args.output), "include_text_ops": args.include_text_ops, "splits": {}}
    for split in ("train", "dev", "test"):
        examples = make_examples(read_rows(args.input / f"{split}.csv"), args.include_text_ops)
        path = args.output / f"{split}.jsonl"
        write_jsonl(path, examples)
        by_task = {}
        for item in examples:
            by_task[item["task"]] = by_task.get(item["task"], 0) + 1
        summary["splits"][split] = {"examples": len(examples), "tasks": by_task, "path": str(path)}
        print(f"{split}_examples: {len(examples)}", by_task)
    (args.output / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"metadata_saved: {args.output / 'metadata.json'}")


if __name__ == "__main__":
    main()
