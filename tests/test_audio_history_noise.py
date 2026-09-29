"""Pinned MiniMind-O shifted-label history corruption contract."""
import pytest
import torch

from trainer.train_audio_multitask import (
    AUDIO_PAD_ID, AUDIO_STOP_ID, MultitaskForward,
    perturb_audio_history, perturb_text_history,
)


def test_shifted_audio_mask_replaces_all_eight_including_pad_and_stop(monkeypatch):
    inputs = torch.full((2, 6, 8), AUDIO_PAD_ID)
    inputs[:, 2, 0] = AUDIO_STOP_ID
    targets = torch.full_like(inputs, -1)
    targets[0, 2, 7] = 3
    targets[0, 3, 0] = AUDIO_STOP_ID
    targets[1, 5, 4] = 9
    originals = inputs.clone(), targets.clone()
    bounds = []

    def noise(high, shape, **kwargs):
        bounds.append(high)
        return torch.full(shape, high - 1, **kwargs)

    monkeypatch.setattr(torch, "randint", noise)
    result = perturb_audio_history(inputs, targets, 1.0)
    expected = inputs.clone()
    expected[0, 1:3] = 2111
    expected[1, 4] = 2111
    assert torch.equal(result, expected)
    assert bounds == [2112]
    assert torch.equal(inputs, originals[0]) and torch.equal(targets, originals[1])


def test_audio_bernoulli_is_per_time_not_per_cell(monkeypatch):
    inputs = torch.full((1, 4, 8), AUDIO_PAD_ID)
    labels = torch.full_like(inputs, 1)
    monkeypatch.setattr(torch, "rand", lambda shape, **kw: torch.tensor([[0.04, 0.06, 0.0, 0.0]]))
    monkeypatch.setattr(torch, "randint", lambda high, shape, **kw: torch.zeros(shape, **kw))
    result = perturb_audio_history(inputs, labels, 0.05)
    assert result[0, [0, 2]].eq(0).all()
    assert result[0, [1, 3]].eq(AUDIO_PAD_ID).all()


def test_text_shift_protects_existing_image_not_target_or_replacement(monkeypatch):
    inputs = torch.tensor([[1, 9, 2, 3, 4, 5]])
    labels = torch.tensor([[-100, -100, 7, 9, 8, -100]])
    snapshots = inputs.clone(), labels.clone()
    monkeypatch.setattr(torch, "randint", lambda high, shape, **kw: torch.full(shape, high - 1, **kw))
    result = perturb_text_history(inputs, labels, 1.0, vocab_size=10, image_token_id=9)
    assert result.tolist() == [[1, 9, 9, 9, 4, 5]]
    assert torch.equal(inputs, snapshots[0]) and torch.equal(labels, snapshots[1])


def test_text_and_audio_use_independent_bernoulli_draws(monkeypatch):
    inputs = torch.full((1, 4), 1)
    audio = torch.full((1, 4, 8), AUDIO_PAD_ID)
    draws = iter([torch.zeros((1, 4)), torch.ones((1, 4))])
    monkeypatch.setattr(torch, "rand", lambda shape, **kw: next(draws))
    monkeypatch.setattr(torch, "randint", lambda high, shape, **kw: torch.zeros(shape, **kw))
    noisy_audio = perturb_audio_history(audio, torch.ones_like(audio), 0.05)
    noisy_text = perturb_text_history(inputs, torch.ones_like(inputs), 0.05, 10)
    assert noisy_audio[:, :-1].eq(0).all()
    assert torch.equal(noisy_text, inputs)


@pytest.mark.parametrize("probability", [-0.1, 1.01, float("nan")])
def test_invalid_probability(probability):
    text = torch.zeros((1, 2), dtype=torch.long)
    audio = torch.zeros((1, 2, 8), dtype=torch.long)
    with pytest.raises(ValueError):
        perturb_audio_history(audio, audio, probability)
    with pytest.raises(ValueError):
        perturb_text_history(text, text, probability, 10)


def test_zero_noise_is_identity_without_rng_consumption():
    text = torch.zeros((1, 2), dtype=torch.long)
    audio = torch.zeros((1, 2, 8), dtype=torch.long)
    state = torch.random.get_rng_state()
    assert perturb_audio_history(audio, audio, 0) is audio
    assert perturb_text_history(text, text, 0, 10) is text
    assert torch.equal(state, torch.random.get_rng_state())
