"""Offline tests: python -m unittest discover -s tests -p 'test_s2s_lora.py' -v."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

from model import ddp_utils
from model.qwen3_adapter import forward_hidden_states, lm_head_module
from trainer.train_speech_to_speech import (
    LORA_TARGETS, SpeechToSpeechLoss, configure_tuning, require_extended_vocab,
    save_standalone, token_accuracy,
)


def tiny(tune="lora", checkpoint=False):
    torch.manual_seed(123)
    config = Qwen3Config(vocab_size=48, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=2, head_dim=8,
                         attention_dropout=0.0, tie_word_embeddings=True)
    model = configure_tuning(Qwen3ForCausalLM(config), tune, r=2, alpha=4, dropout=0)
    if checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    return model


def batch(rank=0):
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]]) + rank * 6
    labels = ids.clone()
    labels[:, :2 + rank] = -100
    return ids, torch.ones_like(ids), labels


def distributed_worker(rank, rendezvous):
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        # Compare real DDP backward/update against one global token-mean batch.
        for tune in ("lora", "full", "embed"):
            lm = tiny(tune, checkpoint=True)
            reference = SpeechToSpeechLoss(tiny(tune, checkpoint=True), 2)
            wrapped = ddp_utils.wrap(SpeechToSpeechLoss(lm, 2))
            opt = torch.optim.SGD(wrapped.parameters(), lr=0.1)
            refopt = torch.optim.SGD(reference.parameters(), lr=0.1)
            for _ in range(2):
                ids, mask, labels = batch(rank)
                opt.zero_grad()
                loss, count = wrapped(ids, mask, labels)
                mean_count = ddp_utils.all_reduce_mean(float(count))
                (loss * count / mean_count).backward()
                refopt.zero_grad()
                combined = [torch.cat([batch(0)[i], batch(1)[i]]) for i in range(3)]
                ref_loss, _ = reference(*combined)
                ref_loss.backward()
                for p, q in zip(wrapped.module.parameters(), reference.parameters()):
                    if p.requires_grad:
                        assert p.grad is not None and q.grad is not None
                        torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4)
                opt.step()
                refopt.step()
                for p, q in zip(wrapped.module.parameters(), reference.parameters()):
                    torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-4)
                    other = p.detach().clone()
                    dist.broadcast(other, 0)
                    torch.testing.assert_close(p, other, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


class S2SLoRATest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_lora_backward_checkpoint_and_standalone(self):
        for checkpoint in (False, True):
            with self.subTest(checkpoint=checkpoint):
                lm = tiny(checkpoint=checkpoint)
                self.assertFalse(lm.get_input_embeddings().weight.requires_grad)
                self.assertFalse(lm_head_module(lm).weight.requires_grad)
                names = [n for n, p in lm.named_parameters() if p.requires_grad]
                self.assertTrue(names)
                self.assertTrue(all("lora_" in n for n in names))
                self.assertEqual({target for target in LORA_TARGETS
                                  if any(f".{target}." in n for n in names)}, set(LORA_TARGETS))
                before = {n: p.detach().clone() for n, p in lm.named_parameters()}
                opt = torch.optim.SGD(lm.parameters(), lr=0.2)
                objective = SpeechToSpeechLoss(lm, 2).train()
                loss, count = objective(*batch())
                self.assertEqual(count, 4)
                loss.backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for n, p in lm.named_parameters() if "lora_B" in n))
                opt.step()
                for n, p in lm.named_parameters():
                    if not p.requires_grad:
                        torch.testing.assert_close(p, before[n], atol=0, rtol=0)
                lm.eval()
                ids, mask, _ = batch()
                with torch.no_grad():
                    logits = lm(input_ids=ids, attention_mask=mask).logits
                    hidden = forward_hidden_states(lm, mask, input_ids=ids)
                    torch.testing.assert_close(lm_head_module(lm)(hidden), logits)
                live = {n: p.detach().clone() for n, p in lm.named_parameters()}
                identities = [id(p) for p in lm.parameters()]
                with tempfile.TemporaryDirectory() as directory:
                    save_standalone(lm, directory)
                    self.assertFalse((Path(directory) / "adapter_config.json").exists())
                    restored = AutoModelForCausalLM.from_pretrained(directory).eval()
                    with torch.no_grad():
                        torch.testing.assert_close(restored(ids).logits, logits, atol=2e-6, rtol=2e-4)
                self.assertEqual(identities, [id(p) for p in lm.parameters()])
                for n, p in lm.named_parameters():
                    torch.testing.assert_close(p, live[n], atol=0, rtol=0)
                objective.train()
                opt.zero_grad()
                objective(*batch())[0].backward()
                opt.step()  # export did not unload adapters or invalidate optimizer

    def test_full_embed_and_vocab_guard(self):
        for tune in ("full", "embed"):
            lm = tiny(tune)
            for n, p in lm.named_parameters():
                self.assertEqual(p.requires_grad, tune == "full" or
                                 n.startswith(("model.embed_tokens", "lm_head")))
            SpeechToSpeechLoss(lm, 2)(*batch())[0].backward()
            require_extended_vocab(lm, SimpleNamespace(total_vocab_size=48))
            with self.assertRaises(ValueError):
                require_extended_vocab(lm, SimpleNamespace(total_vocab_size=49))
            with tempfile.TemporaryDirectory() as directory:
                save_standalone(lm, directory)
                restored = AutoModelForCausalLM.from_pretrained(directory)
                self.assertEqual(restored.config.vocab_size, 48)

    def test_invalid_lora_options(self):
        for r, alpha, dropout in ((0, 4, 0), (2, 0, 0), (2, 4, -0.1), (2, 4, 1)):
            with self.assertRaises(ValueError):
                configure_tuning(tiny("full"), "lora", r, alpha, dropout)

    def test_chunked_accuracy(self):
        lm = tiny().eval()
        ids, mask, labels = batch()
        with torch.no_grad():
            prediction = lm(ids).logits[:, :-1].argmax(-1)
            target = labels[:, 1:]
            expected = (prediction[target != -100] == target[target != -100]).float().mean().item()
        sizes = []
        hook = lm_head_module(lm).register_forward_pre_hook(
            lambda module, inputs: sizes.append(inputs[0].shape[0]))
        try:
            with patch("trainer.train_speech_to_speech.build_audio_batch",
                       return_value=(ids, labels, mask)):
                actual = token_accuracy(SpeechToSpeechLoss(lm, 2), None, None, [None],
                                        torch.device("cpu"), SimpleNamespace(
                                            max_length=6, max_answer_frames=1, loss_chunk=2))
            self.assertAlmostEqual(actual, expected)
            self.assertLessEqual(max(sizes), 2)
        finally:
            hook.remove()

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU gloo unavailable")
    def test_two_rank_global_batch_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_worker,
                     args=("file://" + str(Path(directory) / "rendezvous"),),
                     nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
