"""Offline integration tests for the format-v5 Thinker/Talker contract."""
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from model.qwen3_talker import Qwen3ThinkerTalker
from trainer import train_audio_multitask as trainer


@pytest.fixture(autouse=True)
def limited_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def model_and_tokenizer():
    markers = ["[UNK]", "[PAD]", "[EOS]", "<|audio_start|>",
               "<|audio_end|>", "<|audio_pad|>"]
    vocab = {word: index for index, word in enumerate(
        markers + [f"token{index}" for index in range(58)])}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        eos_token="[EOS]", additional_special_tokens=markers[3:],
    )
    assert len(tokenizer) == 64
    with torch.random.fork_rng():
        torch.manual_seed(123)
        config = Qwen3Config(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=4, num_attention_heads=2,
            num_key_value_heads=1, head_dim=8, max_position_embeddings=64,
            attention_dropout=0.0, pad_token_id=1, eos_token_id=2,
            tie_word_embeddings=False,
        )
        config._attn_implementation = "eager"
        model = Qwen3ThinkerTalker(
            Qwen3ForCausalLM(config), num_talker_layers=4,
            bridge_layer=1, adapter_rank=4,
        )
    return model, tokenizer


@pytest.fixture
def saved_checkpoint(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    with TemporaryDirectory(prefix="thinker-talker-test-") as directory:
        path = Path(directory) / "checkpoint"
        trainer.save_checkpoint(model, tokenizer, path, task="s2a", epoch=2)
        yield path, model, tokenizer


def toy_batch():
    # Two real frames for every codebook, then each delayed STOP feedback.
    codes = torch.arange(16).reshape(8, 2)
    targets, history = trainer._delayed_audio_output(codes)
    audio_targets = torch.tensor([[[-1] * 8] + targets])
    audio_inputs = torch.tensor([[[trainer.AUDIO_PAD_ID] * 8] + history])
    length = audio_targets.shape[1]
    inputs = (torch.arange(length).unsqueeze(0) % 50) + 6
    text_targets = torch.full_like(inputs, -100)
    text_targets[:, 1:4] = torch.tensor([7, 8, 2])
    attention = torch.ones_like(inputs)
    return inputs, text_targets, audio_targets, audio_inputs, attention


def test_actual_checkpoint_roundtrip(saved_checkpoint):
    path, original, tokenizer = saved_checkpoint
    metadata = json.loads((path / "multitask_metadata.json").read_text())
    assert metadata["multitask_format_version"] == 5
    assert metadata["architecture"] == "thinker_talker"
    assert metadata["talker_config"] == {
        "num_talker_layers": 4, "bridge_layer": 1, "adapter_rank": 4,
    }
    assert metadata["task"] == "s2a" and metadata["epoch"] == 2
    restored, restored_tokenizer = trainer.load_multitask_checkpoint(path, torch.device("cpu"))
    assert not restored.training
    assert restored_tokenizer.get_vocab() == tokenizer.get_vocab()
    assert restored_tokenizer.pad_token_id == tokenizer.pad_token_id
    assert restored_tokenizer.eos_token_id == tokenizer.eos_token_id
    expected = original.state_dict()
    actual = restored.state_dict()
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    batch = toy_batch()
    original.eval()
    with torch.no_grad():
        before = original.forward_streams(batch[0], batch[3], attention_mask=batch[4])
        after = restored.forward_streams(batch[0], batch[3], attention_mask=batch[4])
        torch.testing.assert_close(after.text_hidden, before.text_hidden)
        torch.testing.assert_close(after.audio_hidden, before.audio_hidden)
        assert after.past_key_values is None
        torch.testing.assert_close(restored.thinker.lm_head(after.text_hidden),
                                   original.thinker.lm_head(before.text_hidden))
        for q in range(8):
            torch.testing.assert_close(restored.audio_streams.heads[q](after.audio_hidden),
                                       original.audio_streams.heads[q](before.audio_hidden))


@pytest.mark.parametrize("version", [3, 4])
def test_previous_formats_are_rejected(saved_checkpoint, version):
    path, _, _ = saved_checkpoint
    metadata_path = path / "multitask_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["multitask_format_version"] = version
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="format-v5|shared-backbone"):
        trainer.load_multitask_checkpoint(path, torch.device("cpu"))


@pytest.mark.parametrize("missing", ["talker.pt", "multitask_metadata.json"])
def test_missing_checkpoint_component_is_rejected(saved_checkpoint, missing):
    path, _, _ = saved_checkpoint
    (path / missing).rename(path / (missing + ".removed"))
    with pytest.raises(ValueError, match="incomplete|metadata.*required"):
        trainer.load_multitask_checkpoint(path, torch.device("cpu"))


def test_missing_talker_metadata_is_rejected(saved_checkpoint):
    path, _, _ = saved_checkpoint
    metadata_path = path / "multitask_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata["talker_config"]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="talker_config"):
        trainer.load_multitask_checkpoint(path, torch.device("cpu"))


def test_missing_talker_state_key_fails_strict_loading(saved_checkpoint):
    path, _, _ = saved_checkpoint
    state = torch.load(path / "talker.pt", weights_only=True)
    missing_key = next(iter(state))
    del state[missing_key]
    torch.save(state, path / "talker.pt")
    with pytest.raises(RuntimeError, match="Missing key") as error:
        trainer.load_multitask_checkpoint(path, torch.device("cpu"))
    assert missing_key in str(error.value)


def test_checkpoint_overwrite_guard_preserves_every_file(saved_checkpoint):
    path, model, tokenizer = saved_checkpoint
    before = {str(file.relative_to(path)): file.read_bytes()
              for file in path.rglob("*") if file.is_file()}
    with pytest.raises(ValueError, match="refusing to overwrite"):
        trainer.save_checkpoint(model, tokenizer, path, task="tts", epoch=99)
    after = {str(file.relative_to(path)): file.read_bytes()
             for file in path.rglob("*") if file.is_file()}
    assert after == before


def test_training_backward_and_optimizer_partition(model_and_tokenizer):
    model, _ = model_and_tokenizer
    wrapped = trainer.MultitaskForward(model, chunk_size=3).train()
    batch = toy_batch()
    assert len(batch) == 5
    for q in range(8):
        assert (batch[2][..., q] == trainer.AUDIO_STOP_ID).sum() == 1
        assert ((batch[2][..., q] >= 0) &
                (batch[2][..., q] < trainer.AUDIO_STOP_ID)).sum() == 2
    # Mirror the trainer's identity-based two-learning-rate partition.
    audio = list(model.audio_streams.parameters())
    audio_ids = {id(parameter) for parameter in audio}
    thinker = [parameter for parameter in wrapped.parameters()
               if parameter.requires_grad and id(parameter) not in audio_ids]
    thinker_ids = {id(parameter) for parameter in thinker}
    assert thinker_ids == {id(parameter) for parameter in model.thinker.parameters()}
    assert thinker_ids.isdisjoint(audio_ids)
    counts = Counter(id(parameter) for parameter in thinker + audio)
    assert set(counts) == {id(parameter) for parameter in wrapped.parameters()
                           if parameter.requires_grad}
    assert all(count == 1 for count in counts.values())
    optimizer = torch.optim.AdamW([
        {"params": thinker, "lr": 1e-4}, {"params": audio, "lr": 1e-3},
    ])
    loss, _ = wrapped(*batch)
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    for name, component in [("thinker", model.thinker), ("talker", model.audio_streams)]:
        gradients = [parameter.grad for parameter in component.parameters()]
        assert all(gradient is not None for gradient in gradients), name
        assert all(torch.isfinite(gradient).all() for gradient in gradients), name
        assert any(torch.count_nonzero(gradient) for gradient in gradients), name
    for component in [model.thinker.model.layers, model.audio_streams.decoder.layers,
                      model.audio_streams.semantic_projection,
                      model.audio_streams.codec_projection,
                      model.audio_streams.embedding_adapters,
                      model.audio_streams.head_adapters]:
        assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad)
                   for parameter in component.parameters())
    tracked = [model.thinker.lm_head.weight, model.audio_streams.head_base.weight]
    before = [parameter.detach().clone() for parameter in tracked]
    optimizer.step()
    assert all(not torch.equal(previous, parameter)
               for previous, parameter in zip(before, tracked))
