"""Evaluate a chapter-04 Speech-MiniMind checkpoint by generation, not loss.

The script loads the same frozen encoder + projector + tuned MiniMind pipeline
as ``scripts/infer_speech_minimind.py`` and scores generated text against the
reference answers in a JSONL manifest.  It reports aggregate and per-task /
per-source / per-language metrics, optional exact-match metrics, and a CSV of
individual predictions for qualitative inspection.

Example:
    python scripts/evaluate_speech_minimind.py \\
      --data data/speech2text_corpus/splits/val.jsonl \\
      --encoder-type sensevoice --sensevoice-model outputs/sensevoice-small \\
      --projector-checkpoint outputs/04_speech_minimind_sft/projector_epoch_003.pt \\
      --minimind-model outputs/04_speech_minimind_sft/model_epoch_003 \\
      --output outputs/04_speech_minimind_sft/eval_dev_epoch003_projector04 \\
      --limit 200 --batch-size 4
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.frozen_encoder import build_frozen_encoder  # noqa: E402
from model.minimind_adapter import generate_from_speech, load_minimind  # noqa: E402
from model.speech_projector import SpeechProjector  # noqa: E402
from scripts.infer_speech_minimind import SAMPLE_RATE, resolve_audio  # noqa: E402
from dataset.speech_dataset import DEFAULT_SYSTEM_PROMPT  # noqa: E402

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_PUNCT = re.compile(r"[，。！？、；：,.!?;:'\"“”‘’（）()\[\]{}【】《》\s]+")
_NUMBER_CN = re.compile(r"\d+")


def row_audio_path(row: dict) -> str:
    """Accept both the stage-2 ``wav`` field and the legacy ``audio`` field."""
    return str(row.get("audio") or row.get("wav") or "").strip()


def row_instruction(row: dict) -> str:
    """Per-row instruction, falling back to the stage-2 system prompt."""
    return str(row.get("instruction", "")).strip() or DEFAULT_SYSTEM_PROMPT


def normalize_text(text: str, *, language: str = "zh") -> tuple[str, str]:
    """Return (normalised text, scoring unit: "char" or "word")."""
    text = (text or "").strip().lower()
    text = text.replace("\u3000", " ")
    text = _PUNCT.sub(" ", text)
    if language == "zh" or _CJK.search(text):
        return "".join(text.split()), "char"
    return " ".join(text.split()), "word"


def edit_counts(reference: str, hypothesis: str) -> tuple[int, int, int]:
    """Return substitutions, deletions, insertions via Levenshtein DP."""
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    distance = [[0] * cols for _ in range(rows)]
    operation = [["match"] * cols for _ in range(rows)]
    for row in range(1, rows):
        distance[row][0] = row
        operation[row][0] = "delete"
    for col in range(1, cols):
        distance[0][col] = col
        operation[0][col] = "insert"
    for row in range(1, rows):
        for col in range(1, cols):
            if reference[row - 1] == hypothesis[col - 1]:
                distance[row][col] = distance[row - 1][col - 1]
                operation[row][col] = "match"
            else:
                choices = [
                    (distance[row - 1][col - 1] + 1, "substitute"),
                    (distance[row - 1][col] + 1, "delete"),
                    (distance[row][col - 1] + 1, "insert"),
                ]
                distance[row][col], operation[row][col] = min(choices)
    substitutions = deletions = insertions = 0
    row, col = len(reference), len(hypothesis)
    while row or col:
        action = operation[row][col]
        if action == "substitute":
            substitutions += 1
            row, col = row - 1, col - 1
        elif action == "delete":
            deletions += 1
            row -= 1
        elif action == "insert":
            insertions += 1
            col -= 1
        else:
            row, col = row - 1, col - 1
    return substitutions, deletions, insertions


def load_rows(path: Path, limit: int = 0, offset: int = 0, sample_per_group: int = 0) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if offset:
        rows = rows[offset:]
    if sample_per_group:
        groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for row in rows:
            key = (str(row.get("source", "")), str(row.get("task", "")), str(row.get("lang", "")))
            if len(groups[key]) < sample_per_group:
                groups[key].append(row)
        rows = [row for key in sorted(groups) for row in groups[key]]
    if limit:
        rows = rows[:limit]
    return rows


def warn_checkpoint_pairing(projector_path: Path, model_path: Path) -> None:
    """Warn when a tuned MiniMind dir is paired with an out-of-phase projector."""
    model_dir = model_path.parent if model_path.name.startswith("model_") else model_path
    if model_dir.name == "04_speech_minimind_sft" or "04_speech_minimind_sft" in model_dir.parts:
        try:
            if projector_path.parent.resolve() != model_dir.resolve():
                print(
                    "WARNING: chapter-04 model dir is paired with a projector outside "
                    f"{model_dir}; if the model was trained with --tune-projector, use its "
                    "projector_epoch_XXX.pt from the same output directory.",
                    file=sys.stderr,
                )
        except OSError:
            pass


def load_pipeline(args, device: torch.device):
    warn_checkpoint_pairing(args.projector_checkpoint, args.minimind_model)
    encoder = build_frozen_encoder(
        args.encoder_type,
        checkpoint=str(args.encoder_checkpoint) if args.encoder_type == "conformer" else None,
        model_id=(args.sensevoice_model if args.encoder_type == "sensevoice" else args.paraformer_model),
        device=device,
    )
    proj_ckpt = torch.load(args.projector_checkpoint, map_location=device, weights_only=False)
    llm_dim = int(proj_ckpt.get("llm_hidden_size", 768))
    projector = SpeechProjector(encoder.output_dim, llm_dim).to(device)
    projector.load_state_dict(proj_ckpt["projector"])
    projector.eval()
    lm, tokenizer = load_minimind(args.minimind_model, device)
    for p in lm.parameters():
        p.requires_grad_(False)
    lm.eval()
    if args.tune == "lora":
        from peft import PeftModel

        lm = PeftModel.from_pretrained(lm, args.minimind_model, is_trainable=False)
        lm.eval()
    return encoder, projector, lm, tokenizer


def batched(items: list, batch_size: int) -> Iterable[list]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def score_rows(rows: list[dict], args, encoder, projector, lm, tokenizer, device: torch.device) -> tuple[list[dict], dict]:
    predictions: list[dict] = []
    aggregate = {
        "rows": 0,
        "scored_rows": 0,
        "skipped_rows": 0,
        "exact_matches": 0,
        "normalized_matches": 0,
        "reference_units": 0,
        "edit_distance": 0,
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
        "generation_seconds": 0.0,
        "audio_seconds": 0.0,
        "errors": 0,
    }
    grouped: dict[tuple[str, str, str], dict] = defaultdict(lambda: {
        "rows": 0, "exact_matches": 0, "normalized_matches": 0,
        "reference_units": 0, "edit_distance": 0,
    })
    started = time.perf_counter()
    for batch in batched(rows, args.batch_size):
        for row_index, row in enumerate(batch, start=aggregate["rows"] + 1):
            aggregate["rows"] += 1
            row_started = time.perf_counter()
            audio_path = Path(row_audio_path(row))
            if not audio_path.is_absolute():
                candidate = ROOT / audio_path
                audio_path = candidate if candidate.exists() else args.data.parent / audio_path
            try:
                audio, rate = resolve_audio(str(audio_path))
                waveform = torch.from_numpy(audio[None].astype(np.float32)).to(device)
                lengths = torch.tensor([audio.size], dtype=torch.long, device=device)
                with torch.no_grad():
                    acoustic, acoustic_lengths = encoder.encode(waveform, lengths, SAMPLE_RATE)
                    projected = projector(acoustic)
                    projected_lengths = projector.output_lengths(acoustic_lengths).clamp_max(projected.size(1))
                    pred, token_ids = generate_from_speech(
                        lm,
                        tokenizer,
                        projected,
                        projected_lengths,
                        row_instruction(row),
                        device,
                        max_new_tokens=args.max_new_tokens,
                        max_speech_tokens=args.max_speech_tokens,
                        temperature=args.temperature,
                    )
            except Exception as exc:  # noqa: BLE001
                aggregate["errors"] += 1
                predictions.append({
                    "index": aggregate["rows"],
                    "audio": str(audio_path),
                    "instruction": row_instruction(row),
                    "reference": str(row.get("answer", "")),
                    "prediction": "",
                    "error": repr(exc),
                    "task": str(row.get("task", "")),
                    "source": str(row.get("source", "")),
                    "lang": str(row.get("lang", "")),
                })
                continue

            ref, scoring_unit = normalize_text(str(row.get("answer", "")), language=str(row.get("lang", "") or "zh"))
            hyp, _ = normalize_text(pred, language=str(row.get("lang", "") or "zh"))
            sub, delete, insert = edit_counts(ref, hyp)
            edit = sub + delete + insert
            exact = str(row.get("answer", "")).strip() == pred.strip()
            normalized_exact = ref == hyp
            aggregate["scored_rows"] += 1
            aggregate["exact_matches"] += int(exact)
            aggregate["normalized_matches"] += int(normalized_exact)
            aggregate["reference_units"] += len(ref)
            aggregate["edit_distance"] += edit
            aggregate["substitutions"] += sub
            aggregate["deletions"] += delete
            aggregate["insertions"] += insert
            aggregate["generation_seconds"] += time.perf_counter() - row_started
            aggregate["audio_seconds"] += audio.size / rate
            key = (str(row.get("source", "")), str(row.get("task", "")), str(row.get("lang", "")))
            item = grouped[key]
            item["rows"] += 1
            item["exact_matches"] += int(exact)
            item["normalized_matches"] += int(normalized_exact)
            item["reference_units"] += len(ref)
            item["edit_distance"] += edit
            predictions.append({
                "index": aggregate["rows"],
                "audio": str(audio_path),
                "instruction": row_instruction(row),
                "reference": str(row.get("answer", "")),
                "prediction": pred,
                "reference_normalized": ref,
                "prediction_normalized": hyp,
                "exact_match": exact,
                "normalized_match": normalized_exact,
                "error_rate": edit / max(len(ref), 1),
                "scoring_unit": scoring_unit,
                "substitutions": sub,
                "deletions": delete,
                "insertions": insert,
                "edit_distance": edit,
                "task": str(row.get("task", "")),
                "source": str(row.get("source", "")),
                "lang": str(row.get("lang", "")),
                "generated_tokens": len(token_ids),
                "audio_seconds": audio.size / rate,
            })
    metrics = dict(aggregate)
    metrics["exact_match"] = metrics.pop("exact_matches") / max(metrics["scored_rows"], 1)
    metrics["normalized_exact_match"] = metrics.pop("normalized_matches") / max(metrics["scored_rows"], 1)
    metrics["error_rate"] = metrics.pop("edit_distance") / max(metrics["reference_units"], 1)
    metrics["error_rate_unit"] = "char_or_word_per_sample"
    metrics["real_time_factor"] = metrics["generation_seconds"] / max(metrics["audio_seconds"], 1e-9)
    metrics["elapsed_seconds"] = time.perf_counter() - started
    groups = []
    for (source, task, lang), item in sorted(grouped.items()):
        groups.append({
            "source": source,
            "task": task,
            "lang": lang,
            "rows": item["rows"],
            "exact_match": item["exact_matches"] / max(item["rows"], 1),
            "normalized_exact_match": item["normalized_matches"] / max(item["rows"], 1),
            "error_rate": item["edit_distance"] / max(item["reference_units"], 1),
            "reference_units": item["reference_units"],
            "edit_distance": item["edit_distance"],
        })
    return predictions, {"aggregate": metrics, "groups": groups}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="JSONL with wav,answer[,instruction,task,source,lang] (stage-2 corpus)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="evaluate first N rows after --offset; 0 means all")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-per-group", type=int, default=0,
                        help="deterministically take at most N rows per source/task/lang group")
    parser.add_argument("--batch-size", type=int, default=1, help="kept for API symmetry; generation is one sample at a time")
    parser.add_argument("--encoder-type", choices=("sensevoice", "conformer", "paraformer"), default="sensevoice")
    parser.add_argument("--encoder-checkpoint", type=Path, default=Path("outputs/02_acoustic_encoder/tiny_conformer_ctc.pt"))
    parser.add_argument("--sensevoice-model", default="iic/SenseVoiceSmall")
    parser.add_argument("--paraformer-model", default=None)
    parser.add_argument("--projector-checkpoint", type=Path, required=True)
    parser.add_argument("--minimind-model", type=Path, required=True)
    parser.add_argument("--tune", choices=("lora", "full"), default="full")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-speech-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.data, args.limit, args.offset, args.sample_per_group)
    print(f"device={device} rows={len(rows)}", flush=True)

    encoder, projector, lm, tokenizer = load_pipeline(args, device)
    predictions, report = score_rows(rows, args, encoder, projector, lm, tokenizer, device)
    prediction_path = args.output / "predictions.csv"
    fields = [
        "index", "audio", "instruction", "reference", "prediction", "error",
        "reference_normalized", "prediction_normalized", "exact_match", "normalized_match",
        "error_rate", "scoring_unit", "substitutions", "deletions", "insertions", "edit_distance",
        "task", "source", "lang", "generated_tokens", "audio_seconds",
    ]
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(predictions)

    group_path = args.output / "group_metrics.csv"
    with group_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["groups"][0].keys()) if report["groups"] else [])
        if report["groups"]:
            writer.writeheader()
            writer.writerows(report["groups"])

    report["config"] = {
        "data": str(args.data),
        "projector_checkpoint": str(args.projector_checkpoint),
        "minimind_model": str(args.minimind_model),
        "tune": args.tune,
        "limit": args.limit,
        "offset": args.offset,
        "sample_per_group": args.sample_per_group,
        "device": str(device),
        "max_new_tokens": args.max_new_tokens,
        "max_speech_tokens": args.max_speech_tokens,
        "temperature": args.temperature,
        "prediction_csv": str(prediction_path),
        "group_metrics_csv": str(group_path),
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    print("groups:")
    for group in report["groups"]:
        print(group)
    print(f"predictions_saved: {prediction_path}")
    print(f"report_saved: {report_path}")


if __name__ == "__main__":
    main()
