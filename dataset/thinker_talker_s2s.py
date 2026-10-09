"""Direct MiniMind-O A2A parquet, with Qwen3 answer-free format-v5 scaffolding.

Data/augmentation policy follows minimind-o f900448c; speaker/reference columns
are deliberately never read. Labels remain unshifted for multitask_loss.
"""
import io
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from model.speech_input import validate_codes
from scripts.infer_s2a import joint_prompt
from trainer.train_audio_multitask import (
    _delayed_audio_output, AUDIO_PAD_ID, perturb_audio_history, perturb_text_history,
)


SYSTEM_PROMPTS = (
    "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
    "你是minimind，一个小巧但有用的语言模型。",
    "你是一个专业的AI助手，请提供有价值的回答。",
    "你是minimind，请尽力帮助用户解决问题。",
    "你是一个可靠的AI，请给出准确的回答。",
    "You are a helpful AI assistant.",
    "You are minimind, a lightweight intelligent assistant.",
    "You are a friendly chatbot. Please answer the user's questions carefully.",
    "You are a knowledgeable AI. Try your best to provide accurate information.",
    "You are minimind, a small but useful language model.",
)
EMPTY_THINK = "<think>\n\n</think>\n\n"


def pre_processing_chat(conversations):
    """Source 20% system injection; keep caller messages and tool metadata intact."""
    result = [dict(turn) for turn in conversations]
    if (result and not any(turn.get("tools") for turn in result)
            and result[0].get("role") != "system" and random.random() < 0.2):
        result.insert(0, {"role": "system", "content": random.choice(SYSTEM_PROMPTS)})
    return result


def speech_prompt(tokenizer, frame_count, prompt_text="", modality="audio", history=None,
                  *, strip_empty_thinking=False):
    if modality not in ("audio", "text", "audio_text", "text_audio"):
        raise ValueError("invalid input modality")
    if type(frame_count) is not int or frame_count < 0 or (not frame_count and modality != "text"):
        raise ValueError("question frame count must be positive for audio")
    markers = ["<|audio_start|>", "<|audio_pad|>", "<|audio_end|>"]
    vocab = tokenizer.get_vocab()
    if any(marker not in vocab for marker in markers):
        raise ValueError("tokenizer lacks speech markers")
    if any(marker in prompt_text for marker in markers):
        raise ValueError("prompt_text contains speech control tokens")
    audio = markers[0] + markers[1] * frame_count + markers[2]
    content = {"audio": audio, "text": prompt_text, "audio_text": audio + "\n\n" + prompt_text,
               "text_audio": prompt_text + "\n\n" + audio}[modality]
    history = [dict(turn) for turn in history] if history is not None else [{"role": "user", "content": prompt_text}]
    users = [i for i, turn in enumerate(history) if turn["role"] == "user"]
    if not users:
        raise ValueError("conversation requires a user")
    history[users[-1]]["content"] = content
    prefix, trailer = joint_prompt(tokenizer, history, strip_empty_thinking=strip_empty_thinking)
    slots = torch.tensor([token == vocab[markers[1]] for token in prefix], dtype=torch.bool)
    if int(slots.sum()) != (0 if modality == "text" else frame_count):
        raise ValueError("chat template must preserve exactly one audio_pad token per question frame")
    return history, prefix, trailer, slots


class ThinkerTalkerS2SDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048, training=True):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.manifest = str(data_path)
        paths = [p.strip() for p in self.manifest.split(",")]
        if not paths or not all(paths):
            raise ValueError("provide comma-separated parquet paths")
        tables = [pa.Table.from_batches(pq.ParquetFile(p).iter_batches()) for p in paths]
        tables = [t.cast(pa.schema([f.with_type(pa.large_string()) if pa.types.is_string(f.type) else f
                                   for f in t.schema])) for t in tables]
        self.table = pa.concat_tables(tables, promote_options="default")
        if not len(self.table) or "conversations" not in self.table.column_names:
            raise ValueError("nonempty parquet conversations required")
        self.tokenizer, self.max_length, self.training = tokenizer, max_length, training

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        columns = [name for name in ("conversations", "question_audios", "answer_audios")
                   if name in self.table.column_names]
        row = self.table.select(columns).slice(index, 1).to_pylist()[0]
        conversations = row["conversations"]
        if isinstance(conversations, str):
            conversations = json.loads(conversations)
        if not isinstance(conversations, list) or not conversations:
            raise ValueError("nonempty conversations required")
        conversations = [dict(t) for t in conversations]
        if any(not isinstance(t["content"], str) for t in conversations):
            raise ValueError("conversation content must be text")
        assistants = [i for i, t in enumerate(conversations) if t["role"] == "assistant"]
        if not assistants:
            raise ValueError("conversation requires an assistant")
        selected = random.randint(0, len(assistants) - 1) if self.training and len(assistants) > 1 else len(assistants) - 1
        for turn in range(selected, -1, -1):
            chosen = conversations[:assistants[turn] + 1]
            rendered = self.tokenizer.apply_chat_template(chosen, tokenize=True, add_generation_prompt=False)
            if len(rendered) + 100 < self.max_length:
                break
        history, answer = chosen[:-1], chosen[-1]["content"]
        users = [t for t in history if t["role"] == "user"]
        if not users:
            raise ValueError("conversation requires a user before the answer")
        questions, answers = row.get("question_audios") or [], row.get("answer_audios") or []
        question = questions[len(users) - 1] if len(users) <= len(questions) else None
        codes = answers[turn] if turn < len(answers) else None
        if codes:
            # Upstream packs frames in groups of eight and ignores an incomplete tail.
            flat = torch.as_tensor(codes)
            frames = flat.numel() // 8
            codes = validate_codes(flat[:frames * 8].reshape(-1, 8).T) if frames else None
        else:
            codes = None
        waveform = load_waveform(io.BytesIO(question)) if question else None
        return dict(history=history, prompt_text=users[-1]["content"], answer_text=answer,
                    answer_codes=codes, waveform=waveform,
                    tools_present=any(t.get("tools") for t in chosen))


def load_waveform(path):
    import soundfile as sf
    waveform, rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if not waveform.size or not np.isfinite(waveform).all() or rate < 1:
        raise ValueError("input waveform must be nonempty and finite")
    if rate != 16000:
        import librosa
        waveform = librosa.resample(waveform.astype(float), orig_sr=rate, target_sr=16000)
    return torch.from_numpy(np.asarray(waveform, dtype=np.float32))


def augment_waveform(wav, sr=16000):
    """The seven upstream transforms, in source order with source probabilities."""
    from scipy.signal import resample
    wav = np.asarray(wav, dtype=np.float32).copy()
    if random.random() < 0.5:
        wav = resample(wav, max(1, int(len(wav) / random.uniform(0.7, 1.6)))).astype(np.float32)
    if random.random() < 0.3:
        wav = wav + np.random.randn(len(wav)).astype(np.float32) * random.uniform(0.001, 0.01)
    if random.random() < 0.3:
        wav = wav * random.uniform(0.8, 1.2)
    if random.random() < 0.2 and len(wav) > sr:
        start = random.randint(0, len(wav) - sr // 4)
        wav[start:start + sr // 4] = 0
    if random.random() < 0.2:
        k = random.choice([3, 5, 7])
        wav = np.convolve(wav, np.ones(k) / k, mode="same").astype(np.float32)
    if random.random() < 0.3:
        ir_len = int(sr * random.uniform(0.05, 0.2))
        ir = np.random.randn(ir_len).astype(np.float32) * np.exp(-np.linspace(0, 10, ir_len))
        ir[0] = 1.0
        ir /= np.sqrt(np.sum(ir ** 2) + 1e-6)
        wav = np.convolve(wav, ir, mode="same").astype(np.float32)
    if random.random() < 0.2:
        pink = np.cumsum(np.random.randn(len(wav))).astype(np.float32)
        pink /= np.max(np.abs(pink)) + 1e-6
        wav = wav + pink * random.uniform(0.003, 0.015)
    return np.clip(wav, -1.0, 1.0).astype(np.float32)


def augment_fbank(fbank):
    """Source SpecAugment on valid (T,560) LFR features, before the encoder."""
    t, d = fbank.shape
    if random.random() < 0.5:
        width = random.randint(1, min(64, d))
        start = random.randint(0, d - width)
        fbank[:, start:start + width] = 0
    if random.random() < 0.5 and t > 1:
        width = random.randint(1, min(10, t))
        start = random.randint(0, t - width)
        fbank[start:start + width, :] = 0
    return fbank


def prepare_s2s_batch(model, tokenizer, samples, max_length, mixture=False, modality=None):
    present = [i for i, s in enumerate(samples) if s.get("waveform") is not None]
    waves = [samples[i]["waveform"] for i in present]
    if mixture:
        waves = [torch.from_numpy(augment_waveform(w.cpu().numpy())) for w in waves]
    features = model.encode_waveforms(waves, augment=mixture) if waves else []
    features = dict(zip(present, features))
    prepared = []
    for i, sample in enumerate(samples):
        hidden = features.get(i, torch.empty(0, model.encoder.output_dim))
        choice = modality or (random.choices(("audio", "text", "audio_text", "text_audio"),
                                            weights=(4, 2, 2, 2))[0] if mixture else "audio")
        if not hidden.shape[0]:
            choice = "text"
        prepared.append(dict(sample, speech_features=hidden, modality=choice))
    return build_s2s_batch(tokenizer, prepared, max_length, training=mixture)


def scheduled_sampling(batch, tokenizer, probability=0.05):
    batch = dict(batch)
    batch["audio_inputs"] = perturb_audio_history(batch["audio_inputs"], batch["audio_targets"], probability)
    original = batch["input_ids"]
    changed = perturb_text_history(original, batch["text_targets"], probability, len(tokenizer),
                                   tokenizer.get_vocab().get("<|image_pad|>"))
    # Qwen control/scaffold/feature slots cannot become random structural tokens.
    # Preserve source shifted eligibility but veto structural inputs AND replacements.
    structural = torch.tensor(tokenizer.all_special_ids, device=original.device)
    protected = torch.isin(original, structural) | torch.isin(changed, structural) | batch["speech_mask"]
    batch["input_ids"] = torch.where(protected, original, changed)
    return batch


def build_s2s_batch(tokenizer, samples, max_length, *, training=False):
    if not samples or type(max_length) is not int or max_length < 1:
        raise ValueError("nonempty batch and positive max_length required")
    rows = []
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError("tokenizer needs a pad or eos token")
    for sample in samples:
        question = sample["speech_features"]
        if question.ndim != 2 or not torch.isfinite(question).all():
            raise ValueError("finite SenseVoice features required")
        answer = sample.get("answer_text")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("S2S requires nonempty answer_text")
        modality = sample.get("modality", "audio" if len(question) else "text")
        history = sample.get("history")
        if training and not sample.get("tools_present", False):
            # Include final-turn tool metadata in the upstream bypass decision.
            history = pre_processing_chat(history if history is not None else [
                {"role": "user", "content": sample.get("prompt_text", "")}])
        strip_thinking = training and random.random() > 0.2
        rendered_history, prefix, trailer, slots = speech_prompt(
            tokenizer, len(question), sample.get("prompt_text", ""), modality, history,
            strip_empty_thinking=strip_thinking)
        if modality == "text":
            question = question[:0]
        if training:
            # Apply the same decision to the FULL rendered conversation, including
            # historical turns and the final answer. Never subtract guessed token
            # counts from offsets: both scaffolds and full text are retokenized.
            rendered = tokenizer.apply_chat_template(
                rendered_history + [{"role": "assistant", "content": answer}],
                tokenize=False, add_generation_prompt=False)
            if strip_thinking:
                rendered = rendered.replace(EMPTY_THINK, "")
            text = list(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            if (text[:len(prefix)] != prefix or text[-len(trailer):] != trailer
                    or len(text) <= len(prefix) + len(trailer)):
                raise ValueError("final assistant must preserve the answer-free Qwen scaffold")
            answer_ids = text[len(prefix):-len(trailer)]
        else:
            answer_ids = list(tokenizer(answer, add_special_tokens=False)["input_ids"])
            text = prefix + answer_ids + trailer
        if not answer_ids or any(token in set(tokenizer.all_special_ids) for token in answer_ids):
            raise ValueError("answer_text must tokenize to nonempty text without control tokens")
        codes = sample.get("answer_codes")
        targets, feedback = _delayed_audio_output(validate_codes(codes)) if codes is not None else ([], [])
        onset = len(prefix) + 1
        length = min(max_length, max(len(text), onset + len(targets) if targets else 0))
        ids = torch.full((length,), pad, dtype=torch.long)
        ids[:min(len(text), length)] = torch.tensor(text[:length])
        text_targets = torch.full((length,), -100, dtype=torch.long)
        end = min(len(text), length)
        if len(prefix) < end:
            text_targets[len(prefix):end] = torch.tensor(text[len(prefix):end])
        audio_targets = torch.full((length, 8), -1, dtype=torch.long)
        audio_inputs = torch.full((length, 8), AUDIO_PAD_ID, dtype=torch.long)
        count = min(len(targets), max(0, length - onset))
        if count:
            audio_targets[onset:onset + count] = torch.tensor(targets[:count])
            audio_inputs[onset:onset + count] = torch.tensor(feedback[:count])
        mask = torch.zeros(length, dtype=torch.bool)
        mask[:min(len(prefix), length)] = slots[:length]
        question = question[:int(mask.sum())]
        rows.append((ids, text_targets, audio_targets, audio_inputs, mask, question))
    length = max(len(row[0]) for row in rows)
    batch = {
        "input_ids": torch.full((len(rows), length), pad, dtype=torch.long),
        "text_targets": torch.full((len(rows), length), -100, dtype=torch.long),
        "audio_targets": torch.full((len(rows), length, 8), -1, dtype=torch.long),
        "audio_inputs": torch.full((len(rows), length, 8), AUDIO_PAD_ID, dtype=torch.long),
        "speech_mask": torch.zeros((len(rows), length), dtype=torch.bool),
        "attention_mask": torch.zeros((len(rows), length), dtype=torch.long),
        "speech_features": [],
    }
    for i, row in enumerate(rows):
        for key, value in zip(("input_ids", "text_targets", "audio_targets", "audio_inputs", "speech_mask"), row):
            batch[key][i, :len(value)] = value
        batch["attention_mask"][i, :len(row[0])] = 1
        batch["speech_features"].append(row[-1])
    return batch
