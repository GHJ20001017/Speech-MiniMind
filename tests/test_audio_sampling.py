"""Distribution-level sampling regressions against pinned MiniMind-O."""
import torch

from model.audio_sampling import sample_audio, sample_text


def test_audio_temperature_top50_and_occurrence_penalty(monkeypatch):
    logits = torch.full((2112,), -10.0, dtype=torch.float64)
    logits[:60] = torch.linspace(-1, 1, 60, dtype=torch.float64)
    logits[2050] = 0.8
    logits[2051] = -0.2
    before = logits.clone()
    # Only the last three matter. The positive STOP is penalized twice,
    # the negative speaker ID once; ignored old ID59 is the largest score.
    history = [59, 2050, 2050, 2051]
    expected = logits / 0.2
    expected[2050] /= 1.05 ** 2
    expected[2051] *= 1.05
    values, indices = expected.topk(50)
    seen = []

    def multinomial(probabilities, count):
        assert count == 1 and probabilities.shape == (50,)
        seen.append(probabilities)
        return torch.tensor([7])

    monkeypatch.setattr(torch, "multinomial", multinomial)
    token = sample_audio(logits, history)
    assert token == indices[7].item()
    torch.testing.assert_close(seen[0], values.softmax(-1), atol=1e-12, rtol=1e-12)
    assert torch.equal(before, logits)
    assert history == [59, 2050, 2050, 2051]


def test_text_nucleus_retains_crossing_token_and_temperature(monkeypatch):
    distribution = torch.tensor([0.05, 0.70, 0.10, 0.15], dtype=torch.float64)
    logits = distribution.log() * (0.75 + 1e-9)
    original = logits.clone()
    seen = []

    def multinomial(probabilities, count):
        seen.append(probabilities)
        return torch.tensor([2])

    monkeypatch.setattr(torch, "multinomial", multinomial)
    assert sample_text(logits) == 2
    expected = torch.tensor([0, 0.70, 0.10, 0.15], dtype=torch.float64) / 0.95
    torch.testing.assert_close(seen[0], expected)
    assert torch.equal(logits, original)


def test_samplers_are_seed_reproducible_and_not_greedy():
    audio = torch.linspace(-0.1, 0.1, 2112)
    text = torch.linspace(-0.1, 0.1, 32)
    def draws():
        return [(sample_audio(audio, [2049, 2049]), sample_text(text)) for _ in range(12)]
    with torch.random.fork_rng():
        torch.manual_seed(33)
        first = draws()
        torch.manual_seed(33)
        assert first == draws()
    assert len(set(first)) > 1
    assert any(a != audio.argmax() and t != text.argmax() for a, t in first)
