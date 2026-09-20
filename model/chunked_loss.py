"""Memory-lean cross-entropy through the LM head.

Route B appends a ``num_codebooks * codebook_size`` audio block (8 x 2048 on the
Mimi codec) to Qwen3-0.6B's text vocabulary, so the LM has **168,056 output
rows**.  The obvious way to compute the loss::

    logits = model(input_ids=...).logits
    F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)), ...)

materialises ``(batch, length, 168056)`` in float32, which is **5.2 GiB** at the
route-B sequence length (``--batch-size 2`` x 4112 tokens), and the loss path
holds four such tensors at once:

* the logits themselves,
* the ``.contiguous()`` copy that ``view(-1, vocab)`` forces on a sliced tensor,
* the ``log_softmax`` output that backward needs, and
* its gradient.

That is ~21 GiB of the 60 GiB the S2S run had allocated when it died inside
``loss.backward()`` with ``Tried to allocate 3.30 GiB`` on an 80 GB A800 - by far
the largest single consumer in the trainer, and the one that decides whether
``--batch-size 2`` fits.

:func:`chunked_cross_entropy` computes the same number from the decoder's hidden
states instead: the LM head runs on ``chunk_size`` positions at a time and its
output is **recomputed during the backward pass** (activation checkpointing), so
the peak loss memory is one chunk rather than four full-length tensors.  The loss
value and its gradient are identical to the one-shot version - this is a memory
change, not a modelling one.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def use_expandable_segments() -> None:
    """Ask the CUDA caching allocator to grow in segments.

    The same run that OOMed reported ``16.45 GiB is reserved by PyTorch but
    unallocated`` - i.e. the allocator was holding memory it could not hand out,
    because the vocab-sized tensors come and go as a few huge blocks.  Expanding
    in segments removes that fragmentation trap.  Only the default is changed:
    an explicit ``PYTORCH_ALLOC_CONF`` / ``PYTORCH_CUDA_ALLOC_CONF`` in the
    environment wins, so any user setting keeps working.  Must be called before
    the first CUDA allocation.
    """
    import os

    if "PYTORCH_ALLOC_CONF" in os.environ or "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
        return
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def _head_logits(lm_head, hidden_chunk: torch.Tensor) -> torch.Tensor:
    """``lm_head(hidden_chunk)`` without keeping the logits until backward.

    The chunk's logits are exactly what we are trying not to store, so they are
    recomputed in the backward pass.  When nothing in the chunk requires grad
    (dev pass, frozen backbone) there is no backward to recompute for and the
    head just runs normally.
    """
    if not torch.is_grad_enabled() or not hidden_chunk.requires_grad:
        return lm_head(hidden_chunk)
    return checkpoint(lm_head.__call__, hidden_chunk, use_reentrant=False)


def chunked_cross_entropy(
    lm_head,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = 256,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, int]:
    """Next-token CE of ``labels`` given the decoder's ``hidden_states``.

    Equivalent to::

        logits = lm_head(hidden_states)
        F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                        labels[:, 1:].contiguous().view(-1),
                        ignore_index=ignore_index)

    but the vocab-sized activations never exist for more than ``chunk_size``
    positions (``chunk_size <= 0`` puts the whole sequence through the head in a
    single call, i.e. the pre-chunking behaviour).

    Returns ``(loss, supervised_tokens)``.  ``loss`` is the token-mean over the
    supervised positions and is differentiable; ``supervised_tokens`` is a plain
    ``int`` so callers can weight a batch by how much it actually supervises.
    """
    hidden = hidden_states[:, :-1, :]  # position t predicts label t+1
    targets = labels[:, 1:]
    flat_hidden = hidden.reshape(-1, hidden.size(-1))
    flat_targets = targets.reshape(-1)
    supervised = int((flat_targets != ignore_index).sum().item())

    total = torch.zeros((), device=flat_hidden.device, dtype=torch.float32)
    step = flat_hidden.size(0) if chunk_size <= 0 else int(chunk_size)
    for start in range(0, flat_hidden.size(0), step):
        logits = _head_logits(lm_head, flat_hidden[start:start + step])
        total = total + F.cross_entropy(
            logits,
            flat_targets[start:start + step],
            ignore_index=ignore_index,
            reduction="sum",
        )
    return total / max(supervised, 1), supervised
