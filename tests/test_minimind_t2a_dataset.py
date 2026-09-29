"""Contract tests for direct MiniMind Parquet TTS ingestion."""
import builtins
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from dataset.minimind_t2a_dataset import MiniMindT2ADataset, _text_language, _text_split
from dataset.multitask_audio_dataset import MultiTaskAudioSample


CODES = list(range(64))


def selected_text(split="train", prefix="Assistant answer", seed=7):
    return next(f"{prefix} {i}" for i in range(10000) if _text_split(f"{prefix} {i}", seed) == split)


def record(text=None, codes=None):
    text = selected_text() if text is None else text
    return {"conversations": json.dumps([{"role": "user", "content": "Never train on this question"},
                                         {"role": "assistant", "content": text}], ensure_ascii=False),
            "answer_audios": [CODES if codes is None else codes]}


def write_parquet(tmp_path, records, audio_type=None):
    schema = pa.schema([("conversations", pa.string()),
                        ("answer_audios", audio_type or pa.list_(pa.list_(pa.int64())))])
    path = tmp_path / "data.parquet"
    pq.write_table(pa.Table.from_pylist(records, schema=schema), path, row_group_size=2)
    return path


def test_assistant_source_frame_major_and_unchanged_targets(tmp_path):
    text = selected_text()
    path = write_parquet(tmp_path, [record(text, list(range(80)))])
    dataset = MiniMindT2ADataset(path)
    sample = dataset[0]
    assert isinstance(sample, MultiTaskAudioSample)
    assert sample.task == "tts" and sample.text == text and sample.prompt_codes is None
    assert sample.codes.dtype == torch.long and sample.codes.is_contiguous()
    assert torch.equal(sample.codes, torch.arange(80).reshape(10, 8).T)
    assert dataset._codes[0].dtype == np.uint16
    assert dataset.rows == [dict(text=text, lang="en", source_row=0, assistant_index=0)]
    sample.codes.fill_(2050)
    dataset.set_epoch(19)
    assert torch.equal(dataset[0].codes, torch.arange(80).reshape(10, 8).T)
    assert dataset.stats["selected"] == 1


def test_every_assistant_and_ordinal_alignment(tmp_path):
    texts = [selected_text(prefix=f"Reply {i}") for i in range(3)]
    messages = [{"role": "system", "content": "ignore"}]
    for text in texts:
        messages.extend([{"role": "user", "content": "wrong"}, {"role": "assistant", "content": text}])
    row = {"conversations": json.dumps(messages), "answer_audios": [[i] * 64 for i in range(3)]}
    dataset = MiniMindT2ADataset(write_parquet(tmp_path, [row]))
    assert [row["text"] for row in dataset.rows] == texts
    for i in range(3):
        assert dataset.rows[i]["assistant_index"] == i
        assert torch.all(dataset[i].codes == i)


def test_grouped_splits_normalization_and_determinism(tmp_path):
    records = []
    groups = {}
    for split in ("train", "dev", "test"):
        text = selected_text(split)
        variants = [text, "  " + text.upper().replace(" ", "\t  ") + "  ",
                    "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in text)]
        groups[split] = set(variants)
        records.extend(record(variant) for variant in variants)
    path = write_parquet(tmp_path, records)
    seen = []
    for split in groups:
        dataset = MiniMindT2ADataset(path, split=split)
        assert {row["text"] for row in dataset.rows} == groups[split]
        assert dataset.rows == MiniMindT2ADataset(path, split=split).rows
        assert dataset.rows == MiniMindT2ADataset(path, split=split, seed=7).rows
        seen.append({row["source_row"] for row in dataset.rows})
    assert not (seen[0] & seen[1] or seen[1] & seen[2] or seen[0] & seen[2])
    assert any(_text_split(str(i), 7) != _text_split(str(i), 8) for i in range(1000))


def test_limit_exact_language_and_early_stop(tmp_path):
    texts = [selected_text(prefix=prefix) for prefix in ("你好", "hello", "你好 hello", "!!!", "世界")]
    rows = [record(text) for text in texts]
    path = write_parquet(tmp_path, rows + [{"conversations": "broken", "answer_audios": []}])
    dataset = MiniMindT2ADataset(path, lang_filter="zh", limit=2)
    assert [row["text"] for row in dataset.rows] == [texts[0], texts[4]]
    assert dataset.stats == dict(rows_scanned=5, assistants_scanned=5, split_filtered=0,
                                 language_filtered=3, selected=2)
    english = MiniMindT2ADataset(path, lang_filter="en", limit=1)
    assert english.rows[0]["text"] == texts[1]
    assert english.stats["rows_scanned"] == 2
    with pytest.raises(ValueError, match="row 5.*JSON"):
        MiniMindT2ADataset(path)


@pytest.mark.parametrize("text,lang", [("你好", "zh"), ("CAFÉ", "en"), ("你好 A", "mixed"),
                                        ("123!?", "other"), ("かな", "other")])
def test_language_heuristic(text, lang):
    assert _text_language(text) == lang


@pytest.mark.parametrize("codes,error", [([-1] * 64, r"\[0, 2048\)"),
                                          ([2048] * 64, r"\[0, 2048\)"),
                                          ([2050] * 64, r"\[0, 2048\)"),
                                          ([2] * 65, "divisible by 8"),
                                          ([], "nonempty"), ([2] * 8, "below min_frames"),
                                          ([None] * 64, "integers, not null")])
def test_invalid_selected_codes_report_row_and_assistant(tmp_path, codes, error):
    path = write_parquet(tmp_path, [record(codes=codes)])
    with pytest.raises(ValueError, match=f"row 0, assistant 0:.*{error}"):
        MiniMindT2ADataset(path, min_frames=8)


def test_default_preserves_valid_short_audio(tmp_path):
    path = write_parquet(tmp_path, [record(codes=[2] * 8)])
    sample = MiniMindT2ADataset(path)[0]
    assert sample.codes.shape == (8, 1)
    assert sample.codes.eq(2).all()


@pytest.mark.parametrize("bad_audio", [None, []])
def test_null_or_missing_audio(tmp_path, bad_audio):
    row = record()
    row["answer_audios"] = bad_audio
    with pytest.raises(ValueError, match="row 0:.*(null|count mismatch)"):
        MiniMindT2ADataset(write_parquet(tmp_path, [row]))


def test_null_single_audio(tmp_path):
    row = record()
    row["answer_audios"] = [None]
    with pytest.raises(ValueError, match="row 0, assistant 0:.*nonempty"):
        MiniMindT2ADataset(write_parquet(tmp_path, [row]))


@pytest.mark.parametrize("text", ["", " \n\t", None, 123])
def test_invalid_assistant_text(tmp_path, text):
    row = record()
    row["conversations"] = json.dumps([{"role": "assistant", "content": text}])
    with pytest.raises(ValueError, match="row 0, assistant 0:.*nonempty text"):
        MiniMindT2ADataset(write_parquet(tmp_path, [row]))


def test_pairing_checked_even_outside_selected_split(tmp_path):
    row = record(selected_text("dev"))
    row["answer_audios"] = []
    with pytest.raises(ValueError, match="row 0:.*count mismatch"):
        MiniMindT2ADataset(write_parquet(tmp_path, [row]), split="train")


@pytest.mark.parametrize("raw", ["bad json", "{}", '[{"content":"missing role"}]', "[]"])
def test_invalid_conversation(tmp_path, raw):
    row = record()
    row["conversations"] = raw
    with pytest.raises(ValueError, match="row 0"):
        MiniMindT2ADataset(write_parquet(tmp_path, [row]))


@pytest.mark.parametrize("audio_type,codes", [(pa.list_(pa.list_(pa.float64())), [0.5] * 64),
                                             (pa.list_(pa.list_(pa.bool_())), [True] * 64),
                                             (pa.list_(pa.list_(pa.list_(pa.int64()))), [[1] * 8] * 8)])
def test_float_bool_and_nested_shapes_rejected_by_schema(tmp_path, audio_type, codes):
    with pytest.raises(ValueError, match="answer_audios schema"):
        MiniMindT2ADataset(write_parquet(tmp_path, [record(codes=codes)], audio_type))


def test_missing_schema_column(tmp_path):
    path = tmp_path / "missing.parquet"
    pq.write_table(pa.table({"conversations": ["[]"]}), path)
    with pytest.raises(ValueError, match="answer_audios.*column"):
        MiniMindT2ADataset(path)


@pytest.mark.parametrize("kwargs", [{"task": "asr"}, {"task": "audio_lm"}, {"split": "validation"},
                                    {"lang_filter": "mixed"}, {"limit": -1}, {"limit": 1.5},
                                    {"min_frames": 0}, {"seed": "7"}, {"max_seq_len": 0},
                                    {"max_seq_len": 1.5}])
def test_invalid_options(tmp_path, kwargs):
    with pytest.raises(ValueError):
        MiniMindT2ADataset(tmp_path / "not-opened.parquet", **kwargs)


def test_dependency_error_is_actionable(tmp_path, monkeypatch):
    original = builtins.__import__

    def no_arrow(name, *args, **kwargs):
        if name.startswith("pyarrow"):
            raise ImportError("simulated absent dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_arrow)
    with pytest.raises(ImportError, match="python -m pip install pyarrow"):
        MiniMindT2ADataset(tmp_path / "not-opened.parquet")
