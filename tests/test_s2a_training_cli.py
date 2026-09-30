"""S2A-only CLI boundary; shared legacy helpers remain independently tested."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from trainer import train_audio_multitask as training


@pytest.mark.parametrize("task", ["tts", "audio_lm", "asr", "s2a,tts"])
@pytest.mark.parametrize("source", ["--qwen3-model", "--init-checkpoint"])
def test_unsupported_cli_task_has_no_training_side_effects(tmp_path, monkeypatch, capsys, task, source):
    output = tmp_path / "output"
    guards = []
    for name in ("load_qwen3", "load_multitask_checkpoint", "validate_init_checkpoint",
                 "reject_audio_checkpoint", "prepare_stage_output", "existing_stages",
                 "load_stage_dataset", "save_checkpoint"):
        guard = Mock(side_effect=AssertionError(f"unexpected {name}"))
        monkeypatch.setattr(training, name, guard)
        guards.append(guard)
    setup = Mock(side_effect=AssertionError("unexpected distributed setup"))
    monkeypatch.setattr(training.ddp_utils, "setup", setup)
    monkeypatch.setattr(training, "wandb", Mock())
    monkeypatch.setattr(sys, "argv", ["train", "--data", "missing.parquet", source, "missing",
                                    "--output", str(output), "--task", task, "--wandb"])
    with pytest.raises(SystemExit) as error:
        training.main()
    assert error.value.code == 2
    assert "supports only --task s2a" in capsys.readouterr().err
    assert not output.exists()
    for guard in guards:
        guard.assert_not_called()
    setup.assert_not_called()
    training.wandb.init.assert_not_called()


@pytest.mark.parametrize("explicit_task", [False, True])
@pytest.mark.parametrize("warm_start", [False, True])
def test_s2a_cli_default_and_explicit_complete_run(tmp_path, monkeypatch, explicit_task, warm_start):
    """Run orchestration to completion without model downloads or GPU work."""
    output = tmp_path / "output"
    # Existing directories are naming history, not a prerequisite task sequence.
    (output / "stage_01_tts").mkdir(parents=True)
    thinker_talker = torch.nn.Module()
    thinker_talker.thinker = torch.nn.Linear(2, 2)
    thinker_talker.audio_streams = torch.nn.Linear(2, 2)
    thinker_talker.config = SimpleNamespace(max_position_embeddings=1024)
    thinker_talker.talker_config = lambda: {"num_talker_layers": 4, "bridge_layer": 0, "adapter_rank": 1}
    tokenizer = Mock()
    tokenizer.get_vocab.return_value = {}
    # len(tokenizer) is required by the forward wrapper.
    from unittest.mock import MagicMock
    tokenizer = MagicMock(wraps=tokenizer)
    tokenizer.__len__.return_value = 20
    monkeypatch.setattr(training, "reject_audio_checkpoint", Mock())
    monkeypatch.setattr(training, "validate_init_checkpoint", Mock(return_value={
        "talker_config": thinker_talker.talker_config()}))
    fresh_load = Mock(return_value=(thinker_talker, tokenizer))
    warm_load = Mock(return_value=(thinker_talker, tokenizer))
    monkeypatch.setattr(training, "load_qwen3", fresh_load)
    monkeypatch.setattr(training, "load_multitask_checkpoint", warm_load)
    monkeypatch.setattr(training, "Qwen3ThinkerTalker", lambda model, **kwargs: model)
    monkeypatch.setattr(training, "register_audio_special_tokens", Mock())
    monkeypatch.setattr(training, "build_vocab_spec", Mock())
    monkeypatch.setattr(training.ddp_utils, "setup", Mock())
    monkeypatch.setattr(training.ddp_utils, "cleanup", Mock())
    monkeypatch.setattr(training.ddp_utils, "device", lambda: torch.device("cpu"))
    monkeypatch.setattr(training.ddp_utils, "is_main", lambda: True)
    monkeypatch.setattr(training.ddp_utils, "is_distributed", lambda: False)
    monkeypatch.setattr(training.ddp_utils, "make_sampler", Mock(return_value=None))
    dataset = Mock(return_value=[object()])
    monkeypatch.setattr(training, "load_stage_dataset", dataset)
    class Loader:
        dataset = [object()]

        def __len__(self):
            return 1
    monkeypatch.setattr(training, "make_loader", Mock(return_value=Loader()))
    components = {"text_loss": 1.0, "audio_loss": 2.0}
    epoch = Mock(return_value=(3.0, {"joint_loss": 3.0, **components}, components))
    monkeypatch.setattr(training, "run_epoch", epoch)
    save = Mock()
    monkeypatch.setattr(training, "save_checkpoint", save)
    monkeypatch.setattr(training, "wandb", Mock())
    argv = ["train", "--data", "pairs.parquet", "--output", str(output),
            "--epochs", "1", "--lr-schedule", "none", "--wandb",
            "--init-checkpoint" if warm_start else "--qwen3-model", "source"]
    if explicit_task:
        argv += ["--task", "s2a"]
    monkeypatch.setattr(sys, "argv", argv)
    training.main()
    selected, unused = (warm_load, fresh_load) if warm_start else (fresh_load, warm_load)
    selected.assert_called_once()
    unused.assert_not_called()
    assert [call.args[1:3] for call in dataset.call_args_list] == [("s2a", "train"), ("s2a", "dev")]
    assert epoch.call_count == 2
    for call in epoch.call_args_list:
        assert set(call.args[4]) == {"s2a"}
        assert call.kwargs["return_components"] is True
    stage_dir = output / "stage_02_s2a"
    metadata = json.loads((stage_dir / "config.json").read_text())
    assert metadata["task"] == "s2a"
    assert metadata["stage_schedule"].startswith("s2a_only;")
    assert "serial" not in metadata["stage_schedule"]
    assert metadata["dataset"]["s2a_semantics"] == training.S2A_SEMANTICS
    save.assert_called_once_with(thinker_talker, tokenizer, stage_dir / "model_epoch_001", "s2a", 1)
    logged = training.wandb.log.call_args.args[0]
    assert logged["train/s2a/joint_loss"] == 3.0
    assert logged["train/s2a/text_loss"] == 1.0
    assert logged["train/s2a/audio_loss"] == 2.0
    training.ddp_utils.cleanup.assert_called_once()
