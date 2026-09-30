"""Component metrics must not change the joint optimization objective."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from trainer import train_audio_multitask as trainer


@pytest.mark.parametrize("chunk", [0, 1, 3])
@pytest.mark.parametrize("modality", ["both", "audio", "text"])
def test_components_preserve_loss_gradients_and_stop_weight(monkeypatch, chunk, modality):
    torch.manual_seed(11)
    text_head = torch.nn.Linear(3, 7)
    heads = torch.nn.ModuleList([torch.nn.Linear(3, trainer.AUDIO_HEAD_SIZE) for _ in range(8)])
    backbone = SimpleNamespace(thinker=text_head, audio_streams=SimpleNamespace(heads=heads))
    monkeypatch.setattr(trainer, "lm_head_module", lambda model: model)
    hidden = torch.randn(1, 4, 3, requires_grad=True)
    audio_hidden = torch.randn(1, 4, 3, requires_grad=True)
    text = torch.tensor([[-100, 1, 2, -100]])
    audio = torch.full((1, 4, 8), -1, dtype=torch.long)
    audio[:, 1, :2] = 3
    audio[:, 2, :2] = trainer.AUDIO_STOP_ID
    if modality == "audio":
        text.fill_(-100)
    if modality == "text":
        audio.fill_(-1)
    parameters = [hidden, audio_hidden, *text_head.parameters(), *heads.parameters()]
    old_loss, old_count = trainer.multitask_loss(
        backbone, hidden, text, audio, chunk, audio_hidden=audio_hidden)
    old_grads = torch.autograd.grad(old_loss, parameters, allow_unused=True)
    loss, count, components = trainer.multitask_loss(
        backbone, hidden, text, audio, chunk, audio_hidden=audio_hidden, return_components=True)
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    assert count == old_count
    assert torch.equal(loss, old_loss)
    for before, after in zip(old_grads, grads):
        if before is None:
            assert after is None
        else:
            assert torch.equal(before, after)
    assert all(not value.requires_grad and value.grad_fn is None for value in components.values())
    torch.testing.assert_close(loss, components["text_loss"] + components["audio_loss"])
    expected_text = (F.cross_entropy(text_head(hidden[0, :2]), text[0, 1:3])
                     if modality != "audio" else torch.tensor(0.0))
    expected_audio = torch.tensor(0.0)
    if modality != "text":
        for head in heads[:2]:
            ce = F.cross_entropy(head(audio_hidden[0, :2]), audio[0, 1:3, 0], reduction="none")
            expected_audio = expected_audio + (ce[0] + trainer.AUDIO_STOP_WEIGHT * ce[1]) / 2 / 8
    torch.testing.assert_close(components["text_loss"], expected_text)
    torch.testing.assert_close(components["audio_loss"], expected_audio)


class FakeLoader(list):
    def __init__(self):
        super().__init__([torch.tensor(float(x)) for x in (1, 3, 8)])
        self.epochs = []
        self.dataset = SimpleNamespace(set_epoch=self.epochs.append)
        self.sampler = None


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.audio_weight = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(self, value, *, return_components=False):
        self.calls += 1
        text = value * self.weight
        audio = 2 * value * self.audio_weight
        result = (text + audio, 1)
        if return_components:
            return (*result, {"text_loss": text.detach(), "audio_loss": audio.detach()})
        return result


def fake_epoch_setup(monkeypatch, main=True, ema=0.5):
    logs, reduced = [], []
    monkeypatch.setattr(trainer, "build_multitask_batch", lambda tokenizer, spec, samples, length: (samples,))
    monkeypatch.setattr(trainer.ddp_utils, "is_main", lambda: main)
    monkeypatch.setattr(trainer.ddp_utils, "set_epoch", lambda *args: None)
    # Simulate equal-rank averaging with a second rank having 3x local loss.
    def reduce(value):
        reduced.append(value)
        return 2 * value
    monkeypatch.setattr(trainer.ddp_utils, "all_reduce_mean", reduce)
    monkeypatch.setattr(trainer, "wandb", SimpleNamespace(log=logs.append))
    args = SimpleNamespace(max_length=10, grad_accum_steps=2, grad_clip=100., loss_ema=ema, wandb=True)
    model = FakeModel()
    optimizer = torch.optim.SGD([{"params": [model.weight]}, {"params": [model.audio_weight]}], lr=0.)
    return args, model, optimizer, FakeLoader(), logs, reduced


@pytest.mark.parametrize("main", [True, False])
def test_accumulation_partial_tail_distributed_means_and_persistent_ema(monkeypatch, main):
    args, model, optimizer, loader, logs, reduced = fake_epoch_setup(monkeypatch, main)
    loss, emas, components = trainer.run_epoch(
        args, model, None, None, {"s2a": loader}, torch.device("cpu"), optimizer,
        epoch=1, return_components=True)
    assert loss == 24
    assert components == {"text_loss": 8, "audio_loss": 16}
    assert emas == {"joint_loss": 30, "text_loss": 10, "audio_loss": 20}
    assert reduced == [6, 2, 4, 24, 8, 16, 12, 4, 8]
    assert model.calls == 3
    if main:
        assert len(logs) == 2
        assert logs[0]["train/s2a/text_loss_step"] == 4
        assert logs[1]["train/s2a/audio_loss_step"] == 32
        assert logs[1]["train/s2a/joint_loss_step"] == 48
        assert logs[1]["train/s2a/text_loss_ema"] == 10
        assert logs[1]["train/s2a/audio_loss_ema"] == 20
    else:
        assert logs == []
    _, next_emas, _ = trainer.run_epoch(
        args, model, None, None, {"s2a": loader}, torch.device("cpu"), optimizer,
        epoch=2, loss_ema=emas, return_components=True)
    assert next_emas == {"joint_loss": 34.5, "text_loss": 11.5, "audio_loss": 23}
    assert emas == {"joint_loss": 30, "text_loss": 10, "audio_loss": 20}
    assert loader.epochs == [1, 2]


def test_dev_epoch_wandb_payload_and_legacy_api(monkeypatch):
    args, model, optimizer, loader, logs, reduced = fake_epoch_setup(monkeypatch)
    dev_loss, _, components = trainer.run_epoch(
        args, model, None, None, {"s2a": loader}, torch.device("cpu"), return_components=True)
    assert model.calls == 3
    assert not model.training
    assert logs == []
    emas = {"joint_loss": 30, "text_loss": 10, "audio_loss": 20}
    trainer.wandb.log(trainer.epoch_loss_metrics("s2a", 24, dev_loss, emas, components, components))
    for split in ("train", "dev"):
        assert logs[-1][f"{split}/s2a/joint_loss"] == 24
        assert logs[-1][f"{split}/s2a/text_loss"] == 8
        assert logs[-1][f"{split}/s2a/audio_loss"] == 16
    legacy = trainer.run_epoch(args, model, None, None, {"tts": loader}, torch.device("cpu"), optimizer)
    assert legacy == (24, 30)
    assert "train/tts/audio_loss_step" in logs[-1]
    assert "train/tts/text_loss_step" not in logs[-1]
    for task in ("tts", "audio_lm"):
        payload = trainer.epoch_loss_metrics(task, 24, 12, emas, components, components)
        assert payload == {f"train/{task}/audio_loss": 24, f"dev/{task}/audio_loss": 12,
                           f"train/{task}/audio_loss_ema": 30}


def test_disabled_ema_preserves_none(monkeypatch):
    args, model, optimizer, loader, logs, _ = fake_epoch_setup(monkeypatch, ema=0)
    _, emas, _ = trainer.run_epoch(args, model, None, None, {"s2a": loader}, torch.device("cpu"),
                                   optimizer, return_components=True)
    assert emas == dict.fromkeys(("joint_loss", "text_loss", "audio_loss"))
    assert logs[-1]["train/s2a/text_loss_ema"] is None


def test_forward_opt_in_detached_components_single_backbone_call(monkeypatch):
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = torch.nn.Parameter(torch.randn(1, 3, 2))
            self.thinker = torch.nn.Linear(2, 4)
            self.audio_streams = torch.nn.Module()
            self.audio_streams.heads = torch.nn.ModuleList(
                [torch.nn.Linear(2, trainer.AUDIO_HEAD_SIZE) for _ in range(8)])
            self.calls = 0

        def forward_streams(self, *args, **kwargs):
            self.calls += 1
            return SimpleNamespace(text_hidden=self.hidden, audio_hidden=self.hidden)

    monkeypatch.setattr(trainer, "lm_head_module", lambda model: model)
    backbone = Backbone()
    model = trainer.MultitaskForward(backbone, 0, history_noise_prob=0)
    text = torch.tensor([[-100, 1, 2]])
    audio = torch.full((1, 3, 8), -1, dtype=torch.long)
    batch = (torch.zeros_like(text), text, audio, audio, torch.ones_like(text))
    assert len(model(*batch)) == 2
    result = model(*batch, return_components=True)
    assert backbone.calls == 2
    assert result[2]["audio_loss"].item() == 0
    assert all(not component.requires_grad for component in result[2].values())
    result[0].backward()
    assert backbone.hidden.grad is not None
