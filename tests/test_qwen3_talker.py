"""Offline tests of the independent Qwen3ThinkerTalker API only."""
from copy import deepcopy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from model.qwen3_talker import AUDIO_PAD_ID, Qwen3ThinkerTalker


def tiny(**options):
    torch.manual_seed(123)
    config = Qwen3Config(
        num_hidden_layers=4, hidden_size=32, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        vocab_size=64, max_position_embeddings=32, attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    return Qwen3ThinkerTalker(Qwen3ForCausalLM(config), adapter_rank=8, **options)


def inputs(length=5):
    torch.manual_seed(42)
    text = torch.randint(0, 64, (2, length))
    audio = torch.randint(0, 2048, (2, length, 8))
    audio[:, 0] = AUDIO_PAD_ID
    return text, audio


def assert_grad(parameter):
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    assert parameter.grad.abs().sum() > 0


def test_gradients_both_paths_and_audio_only_bridge():
    model = tiny().train()
    text, audio = inputs()
    result = model.forward_streams(text, audio)
    assert result.text_hidden.shape == result.audio_hidden.shape == (2, 5, 32)
    assert result.past_key_values is None
    audio_logits = model.audio_streams(result.audio_hidden)
    assert audio_logits.shape == (2, 5, 8, 2112)
    torch.nn.functional.cross_entropy(audio_logits.flatten(0, 2),
                                     torch.randint(0, 2112, (80,))).backward()
    assert_grad(model.thinker.model.layers[0].self_attn.q_proj.weight)
    assert model.thinker.model.layers[2].self_attn.q_proj.weight.grad is None
    for name, parameter in model.audio_streams.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert_grad(model.audio_streams.embedding_base.weight)
    assert model.audio_streams.embedding_base.weight.grad[AUDIO_PAD_ID].abs().sum() > 0
    for adapter in model.audio_streams.embedding_adapters:
        assert adapter[0].weight.grad[AUDIO_PAD_ID].abs().sum() > 0
    model.zero_grad(set_to_none=True)
    result = model.forward_streams(text, audio)
    logits = model.thinker.lm_head(result.text_hidden)
    torch.nn.functional.cross_entropy(logits.flatten(0, 1), text.flatten()).backward()
    assert_grad(model.thinker.model.layers[-1].self_attn.q_proj.weight)
    assert all(p.grad is None for p in model.audio_streams.parameters())


@torch.no_grad()
def test_audio_cannot_change_text():
    model = tiny().eval()
    text, audio = inputs()
    original = model.forward_streams(text, audio)
    changed = model.forward_streams(text, torch.full_like(audio, 31))
    reference = model.thinker.model(input_ids=text, use_cache=False).last_hidden_state
    torch.testing.assert_close(original.text_hidden, reference, rtol=0, atol=0)
    torch.testing.assert_close(original.text_hidden, changed.text_hidden, rtol=0, atol=0)
    assert not torch.equal(original.audio_hidden, changed.audio_hidden)


@pytest.mark.parametrize("bridge_layer", [None, 3])
@pytest.mark.parametrize("padding", ["none", "left", "right"])
@torch.no_grad()
def test_cached_full_text_and_all_audio_logits(bridge_layer, padding):
    model = tiny(bridge_layer=bridge_layer).eval()
    text, audio = inputs(6)
    mask = torch.ones_like(text)
    if padding == "left":
        mask[0, :2] = 0
    elif padding == "right":
        mask[0, -2:] = 0
    full = model.forward_streams(text, audio, attention_mask=mask)
    caches = None
    texts, audios = [], []
    for index in range(text.shape[1]):
        result = model.forward_streams(text[:, index:index + 1], audio[:, index:index + 1],
                                       attention_mask=mask[:, :index + 1],
                                       past_key_values=caches, use_cache=True)
        caches = result.past_key_values
        assert isinstance(caches, tuple) and len(caches) == 2
        assert caches[0] is not caches[1]
        assert int(caches[0].get_seq_length()) == int(caches[1].get_seq_length()) == index + 1
        texts.append(model.thinker.lm_head(result.text_hidden))
        audios.append(model.audio_streams(result.audio_hidden))
    # Fully masked query outputs are unspecified; compare every real token and
    # every vocabulary/codebook logit, including padded batch members.
    valid = mask.bool()
    torch.testing.assert_close(torch.cat(texts, 1)[valid],
                               model.thinker.lm_head(full.text_hidden)[valid], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(torch.cat(audios, 1)[valid],
                               model.audio_streams(full.audio_hidden)[valid], atol=2e-5, rtol=2e-5)
    for q, head in enumerate(model.audio_streams.heads):
        torch.testing.assert_close(head(full.audio_hidden), model.audio_streams(full.audio_hidden)[..., q, :])


def test_independent_clone_initialization_ownership_and_config():
    model = tiny(num_talker_layers=2)
    assert model.config is model.thinker.config
    assert model.get_input_embeddings() is model.thinker.get_input_embeddings()
    assert model.audio_streams.decoder.config is not model.config
    assert model.audio_streams.decoder.embed_tokens is None
    assert model.talker_config() == dict(num_talker_layers=2, bridge_layer=1, adapter_rank=8)
    for index, (target, source) in enumerate(zip(model.audio_streams.decoder.layers,
                                                model.thinker.model.layers[-2:])):
        assert target.self_attn.layer_idx == index
        for a, b in zip(target.parameters(), source.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert a.data_ptr() != b.data_ptr()
    torch.testing.assert_close(model.audio_streams.decoder.norm.weight, model.thinker.model.norm.weight)
    assert model.audio_streams.decoder.norm.weight.data_ptr() != model.thinker.model.norm.weight.data_ptr()
    thinker = {id(p) for p in model.thinker.parameters()}
    talker = {id(p) for p in model.audio_streams.parameters()}
    assert not thinker & talker
    assert thinker | talker == {id(p) for p in model.parameters()}
    keys = list(model.audio_streams.state_dict())
    assert keys.count("head_base.weight") == 1
    assert not any(key.startswith("heads.") for key in keys)
    clone = Qwen3ThinkerTalker(Qwen3ForCausalLM(deepcopy(model.config)),
                              **model.talker_config(), initialize_from_thinker=False)
    clone.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    clone.eval()
    with torch.no_grad():
        original, copied = model.forward_streams(*inputs()), clone.forward_streams(*inputs())
    torch.testing.assert_close(original.text_hidden, copied.text_hidden, rtol=0, atol=0)
    torch.testing.assert_close(original.audio_hidden, copied.audio_hidden, rtol=0, atol=0)
    for a, b in zip(model.parameters(), clone.parameters()):
        assert a.data_ptr() != b.data_ptr()


def test_learned_pad_fixed_eight_average_gelu_and_scales():
    streams = tiny().audio_streams
    assert streams.embedding_base.padding_idx is None
    assert isinstance(streams.text_scale, torch.nn.Parameter)
    assert isinstance(streams.audio_scale, torch.nn.Parameter)
    assert streams.text_scale.item() == 3.0
    assert streams.audio_scale.item() == 1.0
    absent = torch.full((1, 1, 8), AUDIO_PAD_ID)
    for adapter, head in zip(streams.embedding_adapters, streams.head_adapters):
        assert adapter[0].padding_idx is None
        assert isinstance(adapter[1], torch.nn.GELU)
        assert isinstance(head[1], torch.nn.GELU)
    for audio in (absent, absent.clone()):
        if audio is not absent:
            audio[..., 3] = 7
        ids = audio.masked_fill(audio == -1, AUDIO_PAD_ID)
        expected = sum(streams.embedding_base(ids[..., q]) + streams.embedding_adapters[q](ids[..., q])
                       for q in range(8)) / 8
        torch.testing.assert_close(streams.embed_audio(audio), expected)
    assert streams.embed_audio(absent).abs().sum() > 0


@pytest.mark.parametrize("options", [
    {"num_talker_layers": 0}, {"num_talker_layers": 5}, {"num_talker_layers": True},
    {"bridge_layer": -1}, {"bridge_layer": 4}, {"bridge_layer": True},
    {"initialize_from_thinker": "yes"},
])
def test_invalid_constructor(options):
    with pytest.raises(ValueError):
        tiny(**options)


def test_invalid_rank_and_config():
    model = tiny()
    with pytest.raises(ValueError, match="adapter_rank"):
        Qwen3ThinkerTalker(model.thinker, adapter_rank=0)
    model.config.num_key_value_heads = 3
    with pytest.raises(ValueError, match="query/KV"):
        Qwen3ThinkerTalker(model.thinker)


@pytest.mark.parametrize("bad_id", [-2, -1, 2112, 9000])
def test_invalid_audio_ids(bad_id):
    model = tiny()
    text, audio = inputs()
    audio[0, 1, 2] = bad_id
    with pytest.raises(ValueError, match="audio input IDs"):
        model.forward_streams(text, audio)


def test_invalid_geometry_text_context_and_cache_atomicity():
    model = tiny().eval()
    text, audio = inputs()
    for invalid in [audio[..., :7], audio[:, :-1], audio.float()]:
        with pytest.raises(ValueError):
            model.forward_streams(text, invalid)
    with pytest.raises(ValueError):
        model.forward_streams(text.float(), audio)
    with pytest.raises(ValueError, match="vocabulary"):
        model.forward_streams(text + 64, audio)
    with pytest.raises(ValueError, match="attention_mask"):
        model.forward_streams(text, audio, attention_mask=torch.ones(2, 4))
    with pytest.raises(ValueError, match="binary"):
        model.forward_streams(text, audio, attention_mask=torch.full_like(text, 2))
    with pytest.raises(ValueError, match="context"):
        model.forward_streams(*inputs(33))
    with pytest.raises(ValueError, match="past_key_values"):
        model.forward_streams(text, audio, past_key_values=object(), use_cache=True)
    with torch.no_grad():
        cache = model.forward_streams(text, audio, use_cache=True).past_key_values
    invalid = audio.clone()
    invalid[0, 0, 0] = 2112
    with pytest.raises(ValueError):
        model.forward_streams(text, invalid, past_key_values=cache, use_cache=True)
    assert int(cache[0].get_seq_length()) == int(cache[1].get_seq_length()) == 5
    with pytest.raises(ValueError, match="independent"):
        model.forward_streams(text, audio, past_key_values=(cache[0], cache[0]), use_cache=True)
    with pytest.raises(ValueError, match="requires"):
        model.forward_streams(text, audio, past_key_values=cache)


@pytest.mark.parametrize("bridge_layer", [None, 3])
@pytest.mark.parametrize("reentrant", [False, True])
def test_checkpointing_preserves_audio_bridge_gradient(bridge_layer, reentrant):
    model = tiny(bridge_layer=bridge_layer).train()
    model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant})
    out = model.forward_streams(*inputs())
    model.audio_streams(out.audio_hidden).square().mean().backward()
    assert_grad(model.thinker.model.layers[0].self_attn.q_proj.weight)
