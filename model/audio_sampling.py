"""Sampling contract from MiniMind-O f900448c608318c53314ebf8a947ab05cd8c038e."""
import torch

GENERATION_CONFIG = {
    "text_temperature": 0.75, "text_top_p": 0.9, "text_repeat_penalty": 1.0,
    "audio_temperature": 0.2, "audio_top_k": 50,
    "audio_repeat_penalty": 1.05, "audio_repeat_window": 3,
    "audio_stop": "first_id_ge_2048; continue_sampling_and_feedback_until_termination",
}


def sample_text(logits):
    scores = logits.clone() / (0.75 + 1e-9)
    sorted_scores, indices = torch.sort(scores, descending=True)
    remove = torch.cumsum(torch.softmax(sorted_scores, dim=-1), dim=-1) > 0.9
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    scores[indices[remove]] = -float("inf")
    return int(torch.multinomial(torch.softmax(scores, dim=-1), 1).item())


def sample_audio(logits, history):
    scores = logits.clone() / 0.2
    # Deliberately do not deduplicate: repeated occurrences compound upstream.
    for token in history[-3:]:
        score = scores[token]
        scores[token] = torch.where(score > 0, score / 1.05, score * 1.05)
    values, indices = scores.topk(50)
    return int(indices[torch.multinomial(torch.softmax(values, dim=-1), 1)].item())
