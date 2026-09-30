"""Training entry-point routing for MiniMind-O Parquet input."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from trainer import train_audio_multitask as training


def test_manifest_routing_preserves_emilia_contract(monkeypatch):
    factory = Mock()
    monkeypatch.setattr(training, "EmiliaTaskDataset", factory)
    result = training.load_stage_dataset(Path("cache"), "asr", "dev", limit=4, seed=11)
    factory.assert_called_once_with(Path("cache/dev.jsonl"), "asr", None,
                                    lang_filter=None, limit=4, seed=11)
    assert result is factory.return_value


def test_parquet_routing_uses_shared_adapter(monkeypatch):
    from dataset import minimind_t2a_dataset

    factory = Mock()
    monkeypatch.setattr(minimind_t2a_dataset, "MiniMindT2ADataset", factory)
    result = training.load_stage_dataset(Path("pairs.parquet"), "tts", "dev",
                                         lang_filter="en", limit=3, seed=9)
    factory.assert_called_once_with(Path("pairs.parquet"), task="tts", split="dev", lang_filter="en", limit=3,
                                    seed=9, tokenizer=None, max_seq_len=training.DEFAULT_S2A_MAX_SEQ_LEN)
    assert result is factory.return_value


@pytest.mark.parametrize("extra, message", [
    (["--task", "asr"], "only --task s2a"),
    (["--task", "s2a", "--limit", "-1"], "nonnegative"),
    (["--task", "s2a", "--dev-limit", "-2"], "nonnegative"),
])
def test_invalid_parquet_cli_fails_before_model_loading(monkeypatch, capsys, extra, message):
    import sys

    load = Mock()
    monkeypatch.setattr(training, "load_qwen3", load)
    monkeypatch.setattr(sys, "argv", ["train_audio_multitask.py", "--data", "pairs.parquet",
                                      "--qwen3-model", "base", *extra])
    with pytest.raises(SystemExit):
        training.main()
    assert message.lower() in capsys.readouterr().err.lower()
    load.assert_not_called()
