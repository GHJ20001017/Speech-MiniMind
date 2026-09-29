"""CPU regressions for full-length Emilia samples and multitask batching."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from dataset.multitask_audio_dataset import EmiliaTaskDataset
from trainer.train_audio_multitask import AUDIO_STOP_ID, build_multitask_batch
from types import SimpleNamespace


class TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        # One token per character keeps the fixture intentionally longer than
        # the old 384-position assumptions without needing a real tokenizer.
        return {"input_ids": [10 + (index % 1000) for index, _ in enumerate(text)]}


def full_codes(frames=600):
    return (torch.arange(8 * frames, dtype=torch.long).reshape(8, frames) % 2048)


class AudioMultitaskFullLengthTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TinyTokenizer()
        self.spec = SimpleNamespace(audio_bos_id=3, audio_eos_id=4)
        self.text = "long transcript " * 40
        self.codes = full_codes()

    def _manifest_dataset(self, task):
        temporary = tempfile.TemporaryDirectory(prefix="audio-full-length-")
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        np.save(directory / "codes.npy", self.codes.numpy())
        (directory / "manifest.jsonl").write_text(
            json.dumps({"codes": "codes.npy", "text": self.text, "lang": "en"}) + "\n",
            encoding="utf-8",
        )
        # max_frames=128 is deliberately passed to exercise the legacy API.
        return EmiliaTaskDataset(
            directory / "manifest.jsonl", task, max_frames=128, min_frames=8,
            continuation_prefix_ratio=(0.25, 0.75), seed=19,
        )

    def test_legacy_max_frames_does_not_crop_asr_or_tts(self):
        for task in ("asr", "tts"):
            with self.subTest(task=task):
                sample = self._manifest_dataset(task)[0]
                self.assertEqual(sample.text, self.text)
                torch.testing.assert_close(sample.codes, self.codes)
                self.assertEqual(sample.codes.shape, (8, 600))

    def test_continuation_prefix_and_suffix_reconstruct_full_recording_each_epoch(self):
        dataset = self._manifest_dataset("audio_lm")
        seen_splits = set()
        for epoch in range(5):
            dataset.set_epoch(epoch)
            sample = dataset[0]
            self.assertIsNotNone(sample.prompt_codes)
            self.assertEqual(sample.text, self.text)
            self.assertGreater(sample.prompt_codes.size(1), 0)
            self.assertGreater(sample.codes.size(1), 0)
            reconstructed = torch.cat((sample.prompt_codes, sample.codes), dim=1)
            torch.testing.assert_close(reconstructed, self.codes)
            seen_splits.add(sample.prompt_codes.size(1))
        self.assertGreater(len(seen_splits), 1)

    def test_long_tts_and_continuation_batches_fit_context_without_truncation(self):
        samples = [
            EmiliaTaskDataset(
                self._manifest_dataset(task).manifest, task, max_frames=128,
                min_frames=8, seed=19,
            )[0]
            for task in ("tts", "audio_lm")
        ]
        target = self.tokenizer(self.text)["input_ids"]
        for sample in samples:
            prompt_frames = 0 if sample.prompt_codes is None else sample.prompt_codes.size(1)
            with self.subTest(task=sample.task):
                inputs, text, audio, audio_inputs, mask = build_multitask_batch(
                    self.tokenizer, self.spec, [sample], max_length=40960,
                )
                length = int(mask[0].sum())
                self.assertGreater(length, 384)
                self.assertEqual(length, inputs.size(1))
                self.assertTrue(torch.equal(mask[0, :length], torch.ones(length, dtype=torch.long)))
                self.assertTrue((mask[0, length:] == 0).all())
                self.assertTrue((text[0, length:] == -100).all())
                self.assertTrue((audio_inputs[0, length:] == 2049).all())
                self.assertEqual(text.shape, inputs.shape)
                self.assertEqual(audio.shape, (1, length, 8))
                self.assertEqual(audio_inputs.shape, (1, length, 8))
                if sample.task == "tts":
                    self.assertEqual(length, len(target) + 600 + 10)
                    self.assertEqual(inputs[0, 1:1 + len(target)].tolist(), target)
                    self._assert_delayed_output(audio[0], audio_inputs[0], 2 + len(target), sample.codes)
                else:
                    self.assertEqual(length, prompt_frames + sample.codes.size(1) + 12)
                    torch.testing.assert_close(
                        audio_inputs[0, 2:2 + prompt_frames], sample.prompt_codes.T,
                    )
                    start = prompt_frames + 4
                    self.assertTrue((audio[0, :start] == -1).all())
                    self._assert_delayed_output(audio[0], audio_inputs[0], start, sample.codes)

    def _assert_delayed_output(self, targets, feedback, start, codes):
        frames = codes.size(1)
        self.assertEqual(targets.size(0) - start, frames + 8)
        for q in range(8):
            first, stop = start + q, start + frames + q
            self.assertTrue((targets[start:first, q] == -1).all())
            torch.testing.assert_close(targets[first:stop, q], codes[q])
            self.assertEqual(targets[stop, q].item(), AUDIO_STOP_ID)
            self.assertTrue((targets[stop + 1:, q] == -1).all())
            self.assertTrue((feedback[start:first, q] == 2049).all())
            torch.testing.assert_close(feedback[first:stop, q], codes[q])
            self.assertEqual(feedback[stop, q].item(), AUDIO_STOP_ID)
            self.assertTrue((feedback[stop + 1:, q] == 2049).all())
        self.assertEqual(int((targets[start:] == AUDIO_STOP_ID).sum()), 8)
        self.assertEqual(feedback[-1].tolist(), [2049] * 7 + [AUDIO_STOP_ID])

    def test_long_continuation_preserves_prompt_and_delays_only_output(self):
        sample = self._manifest_dataset("audio_lm")[0]
        prompt_frames = sample.prompt_codes.size(1)
        expected_length = 1 + prompt_frames + 3 + sample.codes.size(1) + 8
        inputs, text, audio, feedback, mask = build_multitask_batch(
            self.tokenizer, self.spec, [sample], max_length=expected_length,
        )
        self.assertEqual(int(mask.sum()), expected_length)
        self.assertEqual(inputs.shape, (1, expected_length))
        self.assertTrue((text == -100).all())
        torch.testing.assert_close(feedback[0, 2:2 + prompt_frames], sample.prompt_codes.T)
        start = prompt_frames + 4
        self.assertTrue((audio[0, :start] == -1).all())
        self.assertEqual(inputs[0, start - 2:start].tolist(), [4, 3])
        self._assert_delayed_output(audio[0], feedback[0], start, sample.codes)
        with self.assertRaisesRegex(ValueError, "model context limit exceeded"):
            build_multitask_batch(self.tokenizer, self.spec, [sample], expected_length - 1)

    def test_tts_context_guard_includes_all_delay_tail_and_stop_positions(self):
        sample = self._manifest_dataset("tts")[0]
        expected_length = 2 + len(self.tokenizer(sample.text)["input_ids"]) + 600 + 8
        batch = build_multitask_batch(self.tokenizer, self.spec, [sample], expected_length)
        self.assertEqual(batch[0].size(1), expected_length)
        with self.assertRaisesRegex(ValueError, "model context limit exceeded"):
            build_multitask_batch(self.tokenizer, self.spec, [sample], expected_length - 1)

    def test_context_guard_rejects_only_true_small_overrun(self):
        sample = self._manifest_dataset("tts")[0]
        with self.assertRaisesRegex(ValueError, "model context limit exceeded"):
            build_multitask_batch(self.tokenizer, self.spec, [sample], max_length=384)


if __name__ == "__main__":
    unittest.main()
