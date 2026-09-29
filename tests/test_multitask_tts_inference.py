"""CPU-only inference contract tests; no checkpoint downloads or audio models."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from model.qwen3_talker import Qwen3ThinkerTalker
from scripts.evaluate_multitask_tts import synthesize
from scripts.infer_multitask_tts import resolve_config


class Tokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": [3, 4]}


class InferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.tokenizer = Tokenizer()
        self.spec = SimpleNamespace(audio_bos_id=5, audio_eos_id=6)
        thinker = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=32,
        )).eval()
        self.model = Qwen3ThinkerTalker(
            thinker, num_talker_layers=1, initialize_from_thinker=False
        ).eval()

    def _install_heads(self, heads):
        """Route ``audio_streams.heads`` to per-codebook callables (new contract)."""
        self.model.audio_streams.project = lambda hidden, codebook: heads[codebook](hidden)
        return heads

    def test_cached_generation_matches_full_sequence_with_sampled_feedback(self):
        calls = [[] for _ in range(8)]
        original = self.model.audio_streams.project
        batches = []
        forward = self.model.forward_streams

        def recording_forward(inputs, audio, **kwargs):
            batches.append((inputs.clone(), audio.clone()))
            return forward(inputs, audio, **kwargs)

        def recording_project(hidden, codebook):
            logits = original(hidden, codebook)
            calls[codebook].append(logits.detach().clone())
            return logits

        self.model.forward_streams = recording_forward
        self.model.audio_streams.project = recording_project
        codes, votes = synthesize(self.model, self.tokenizer, self.spec, "测试", 3, "cpu")
        self.assertEqual(codes.size(0), 8)
        self.assertLessEqual(codes.size(1), 3)
        self.assertTrue(((codes >= 0) & (codes < 2048)).all())
        inputs = torch.cat([item[0] for item in batches], dim=1)
        audio = torch.cat([item[1] for item in batches], dim=1)
        with torch.inference_mode():
            hidden = forward(inputs, audio, attention_mask=torch.ones_like(inputs),
                             use_cache=False).audio_hidden
            for q in range(8):
                for frame, logits in enumerate(calls[q]):
                    expected = original(hidden[0, 3 + frame + q], q)
                    torch.testing.assert_close(logits[0], expected, atol=1e-6, rtol=1e-5)
        self.assertEqual(len(votes), len(batches))

    def test_cap_and_invalid_budget(self):
        self.model.config.max_position_embeddings = 14
        with self.assertRaisesRegex(ValueError, "refusing to truncate"):
            synthesize(self.model, self.tokenizer, self.spec, "测试", 100, "cpu")
        with self.assertRaisesRegex(ValueError, "positive"):
            synthesize(self.model, self.tokenizer, self.spec, "测试", 0, "cpu")
        self.model.config.max_position_embeddings = 5
        with self.assertRaisesRegex(ValueError, "no room"):
            synthesize(self.model, self.tokenizer, self.spec, "测试", 10, "cpu")

    def test_specials_stop_retained_audio_independently(self):
        class ConstantHead(torch.nn.Module):
            def __init__(self, stop):
                super().__init__()
                self.stop = stop

            def forward(self, hidden):
                logits = hidden.new_zeros(hidden.shape[:-1] + (2112,))
                logits[..., 17] = 100
                if self.stop:
                    logits[..., 2050] = 200
                return logits

        self._install_heads([ConstantHead(True) for _ in range(8)])
        codes, votes = synthesize(self.model, self.tokenizer, self.spec, "测试", 3, "cpu")
        self.assertEqual(tuple(codes.shape), (8, 0))
        self.assertEqual(votes, list(range(1, 9)))
        self._install_heads([ConstantHead(i != 7) for i in range(8)])
        codes, votes = synthesize(self.model, self.tokenizer, self.spec, "测试", 3, "cpu")
        self.assertEqual(tuple(codes.shape), (8, 0))
        self.assertEqual(votes, [1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7])

    def test_delayed_streams_flush_and_crop_inconsistent_stops(self):
        class ScheduledHead(torch.nn.Module):
            def __init__(self, q, length):
                super().__init__()
                self.q, self.length, self.calls = q, length, 0

            def forward(self, hidden):
                logits = hidden.new_full(hidden.shape[:-1] + (2112,), -100.0)
                token = 2050 if self.calls == self.length else 100 * self.q + self.calls
                logits[..., token] = 100.0
                self.calls += 1
                return logits

        for lengths in ([2] * 8, [2, 3, 2, 3, 2, 3, 2, 3]):
            heads = [ScheduledHead(q, length) for q, length in enumerate(lengths)]
            self._install_heads(heads)
            feedback = []
            forward = self.model.forward_streams

            def capture(inputs, audio, **kwargs):
                feedback.append(audio.clone())
                return forward(inputs, audio, **kwargs)

            with patch.object(self.model, "forward_streams", side_effect=capture):
                codes, votes = synthesize(self.model, self.tokenizer, self.spec, "测试", 5, "cpu")
            self.assertEqual(feedback[3][0, 0, 0].item(), 2050)
            self.assertEqual(feedback[4][0, 0, 0].item(), 3)
            expected = torch.tensor([[100 * q, 100 * q + 1] for q in range(8)])
            torch.testing.assert_close(codes, expected)
            self.assertEqual(votes[-1], 8)
            self.assertEqual([h.calls for h in heads], [len(votes) - q for q in range(8)])
            self.assertEqual(len(votes), max(q + n for q, n in enumerate(lengths)) + 1)

    def test_metadata_resolves_training_codec_and_rejects_mismatch(self):
        import json
        from pathlib import Path
        stage = {"task": "tts", "data": "data/train_cache"}
        cache = {"codec_type": "mimi", "codec_model": "/models/mimi", "num_codebooks": 8,
                 "codebook_size": 2048, "sample_rate": 24000, "input_sample_rate": 24000,
                 "frame_rate_hz": 12.5}
        with patch("scripts.infer_multitask_tts.validate_init_checkpoint", return_value={"task": "tts"}), patch.object(
            Path, "read_text", side_effect=lambda: ""
        ) as read:
            read.side_effect = [json.dumps(stage), json.dumps(cache)]
            data, codec, _, _ = resolve_config(Path("stage/model_epoch_003"), None, None)
            self.assertEqual(data, Path(stage["data"]))
            self.assertEqual(codec, "/models/mimi")
            read.side_effect = [json.dumps(stage), json.dumps(cache)]
            with self.assertRaisesRegex(ValueError, "must match"):
                resolve_config(Path("stage/model_epoch_003"), None, "/other/mimi")
            read.side_effect = [json.dumps(stage), json.dumps(cache | {"sample_rate": 16000})]
            with self.assertRaisesRegex(ValueError, "sample_rate"):
                resolve_config(Path("stage/model_epoch_003"), None, None)
        with patch("scripts.infer_multitask_tts.validate_init_checkpoint", return_value={"task": "asr"}):
            with self.assertRaisesRegex(ValueError, "TTS checkpoint"):
                resolve_config(Path("stage/model_epoch_003"), None, None)


if __name__ == "__main__":
    unittest.main()
