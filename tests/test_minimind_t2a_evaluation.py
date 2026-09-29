"""Direct parquet evaluation contracts; CPU-only, no model downloads."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from dataset.minimind_t2a_dataset import MiniMindT2ADataset, _text_split
from scripts import evaluate_multitask_tts as evaluate
from scripts import infer_multitask_tts as infer


def make_parquet(tmp_path, seed=31, split="test"):
    texts = [f"测试 sample {i}" for i in range(5000) if _text_split(f"测试 sample {i}", seed) == split][:3]
    assert len(texts) == 3
    flat = np.arange(96).tolist()
    path = tmp_path / "sft_t2a.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"conversations": json.dumps([{"role": "assistant", "content": text}]),
         "answer_audios": [flat]} for text in texts
    ]), path)
    return path, texts


def stage_config(tmp_path, data):
    stage = {"task": "tts", "data": str(data), "seed": 31}
    (tmp_path / "config.json").write_text(json.dumps(stage))
    return tmp_path / "model_epoch_001"


def test_parquet_requires_explicit_codec_and_declares_uncertainty(tmp_path):
    checkpoint = stage_config(tmp_path, tmp_path / "not_opened.parquet")
    with patch.object(infer, "validate_init_checkpoint", return_value={"task": "tts"}):
        with pytest.raises(ValueError, match="explicit --mimi-model"):
            infer.resolve_config(checkpoint, None, None)
        _, codec, stage, metadata = infer.resolve_config(checkpoint, None, "/chosen/mimi")
    assert codec == "/chosen/mimi" and stage["seed"] == 31
    assert metadata["original_encoder_identity_verified"] is False
    assert "upstream-declared" in metadata["metadata_source"]
    assert metadata["frame_rate_hz"] == 12.5


def test_shared_selection_matches_training_and_bounds_materialization(tmp_path):
    path, texts = make_parquet(tmp_path)
    rows, manifest = evaluate.select_tts_rows(path, "test.jsonl", seed=31, limit=2)
    training_view = MiniMindT2ADataset(path, split="test", seed=31, limit=2)
    assert len(rows) == 2 and manifest == path
    assert [row["text"] for row in rows] == texts[:2]
    for i, row in enumerate(rows):
        torch.testing.assert_close(evaluate.load_codes(row, manifest), training_view[i].codes)
        torch.testing.assert_close(row["codes"], torch.arange(96).reshape(12, 8).T)
        assert row["source_row"] == i


def test_jsonl_relative_reference_unchanged(tmp_path):
    np.save(tmp_path / "codes.npy", np.arange(96).reshape(8, 12))
    (tmp_path / "dev.jsonl").write_text(json.dumps({"text": "hello", "codes": "codes.npy"}) + "\n")
    rows, manifest = evaluate.select_tts_rows(tmp_path, limit=1)
    torch.testing.assert_close(evaluate.load_codes(rows[0], manifest), torch.arange(96).reshape(8, 12))


def test_evaluator_reports_filtered_counts_and_stage_seed(tmp_path):
    path, _ = make_parquet(tmp_path)
    checkpoint = stage_config(tmp_path, path)
    output = tmp_path / "evaluation"
    codec = SimpleNamespace(num_codebooks=8, codebook_size=2048, sample_rate=24000, frame_rate_hz=12.5)
    argv = ["evaluate", "--checkpoint", str(checkpoint), "--data", str(path),
            "--output", str(output), "--mimi-model", "/chosen/mimi", "--device", "cpu",
            "--split", "test", "--codebook-samples", "2", "--num-samples", "1"]
    with patch("sys.argv", argv), \
         patch.object(evaluate, "load_multitask_checkpoint", return_value=(None, None)), \
         patch.object(evaluate, "build_vocab_spec", return_value=None), \
         patch.object(evaluate, "build_frozen_audio_codec", return_value=codec), \
         patch.object(evaluate, "synthesize", return_value=(torch.ones(8, 2, dtype=torch.long), [8])), \
         patch.object(evaluate, "save_wav", return_value=0.96) as save, \
         patch.object(evaluate, "teacher_forced") as teacher:
        evaluate.main()
    report = json.loads((output / "metrics.json").read_text())
    assert report["seed"] == 31 and report["split"] == "test"
    assert report["selection"] == {
        "loaded_rows": 2, "limit": 2, "complete_split_count_known": False,
        "teacher_requested": 2, "teacher_scored": 0,
        "synthesis_requested": 1, "synthesis_scored": 1,
    }
    teacher.assert_not_called()  # 12-frame targets are below the default 40-frame filter.
    torch.testing.assert_close(save.call_args_list[0].args[1], torch.arange(96).reshape(12, 8).T)
    assert report["codec_metadata"]["original_encoder_identity_verified"] is False


@pytest.mark.parametrize("explicit_text", [False, True])
def test_infer_parquet_reference_and_explicit_text(tmp_path, explicit_text):
    path, texts = make_parquet(tmp_path)
    checkpoint = stage_config(tmp_path, path)
    output = tmp_path / "listen"
    codec = SimpleNamespace(num_codebooks=8, codebook_size=2048, sample_rate=24000, frame_rate_hz=12.5)
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=100))
    class Tokenizer:
        bos_token_id = 1
        def __call__(self, text, **kwargs):
            return {"input_ids": [3, 4]}
    argv = ["infer", "--checkpoint", str(checkpoint), "--output", str(output),
            "--mimi-model", "/chosen/mimi", "--device", "cpu", "--split", "test", "--num-samples", "1"]
    if explicit_text:
        argv += ["--text", "custom text"]
    with patch("sys.argv", argv), patch.object(infer, "validate_init_checkpoint", return_value={"task": "tts"}), \
         patch.object(infer, "load_multitask_checkpoint", return_value=(model, Tokenizer())), \
         patch.object(infer, "build_vocab_spec", return_value=None), \
         patch.object(infer, "build_frozen_audio_codec", return_value=codec), \
         patch.object(infer, "synthesize", return_value=(torch.ones(8, 2, dtype=torch.long), [8])), \
         patch.object(infer, "save_wav", return_value=0.96) as save:
        infer.main()
    report = json.loads((output / "metrics.json").read_text())
    assert report["seed"] == 31
    assert report["selection"]["selected_rows"] == 1
    if explicit_text:
        assert save.call_count == 1
        assert report["samples"][0]["text"] == "custom text"
        assert report["split"] is None
    else:
        assert save.call_count == 2
        reference = save.call_args_list[1].args
        torch.testing.assert_close(reference[1], torch.arange(96).reshape(12, 8).T)
        assert reference[2] == output / "sample_00" / "reference.wav"
        assert report["samples"][0]["text"] == texts[0]
        assert report["split"] == "test"
