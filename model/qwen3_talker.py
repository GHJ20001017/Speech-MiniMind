"""Independent text Thinker and audio Talker using public Qwen3 decoders.

Audio inputs are [batch, time, 8], with PAD2049 denoting an inactive stream.
Codec IDs are 0..2047; STOP2050 is fed back; input/output vocabularies are 2112.
The default path keeps the Thinker text-only. Optional explicitly masked speech
embeddings condition its prefix independently of generated answer audio feedback.
"""
from __future__ import annotations

from copy import deepcopy
from functools import partial
from types import SimpleNamespace

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3RMSNorm


NUM_CODEBOOKS = 8
CODEBOOK_SIZE = 2048
AUDIO_STOP_ID = 2050
AUDIO_PAD_ID = 2049
AUDIO_SPEAKER_ID = 2051
AUDIO_INPUT_VOCAB_SIZE = 2112
AUDIO_OUTPUT_VOCAB_SIZE = 2112
_INTEGER_DTYPES = (torch.int32, torch.int64)


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _projection(hidden_size, eps):
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size), nn.GELU(),
        nn.Linear(hidden_size, hidden_size), Qwen3RMSNorm(hidden_size, eps=eps),
    )


class TalkerModule(nn.Module):
    """All new audio parameters, registered exactly once under this module."""

    def __init__(self, thinker_model, num_layers, adapter_rank, initialize):
        super().__init__()
        config = deepcopy(thinker_model.config)
        config.num_hidden_layers = num_layers
        config.vocab_size = AUDIO_INPUT_VOCAB_SIZE
        # Text pad IDs need not be in the much smaller audio vocabulary.
        config.pad_token_id = AUDIO_PAD_ID
        if getattr(config, "layer_types", None) is not None:
            config.layer_types = list(config.layer_types[-num_layers:])
        self.decoder = Qwen3Model(config)
        self.decoder.embed_tokens = None  # inputs_embeds exclusively; no dead table
        hidden = config.hidden_size
        self.embedding_base = nn.Embedding(AUDIO_INPUT_VOCAB_SIZE, hidden)
        self.embedding_adapters = nn.ModuleList([
            nn.Sequential(nn.Embedding(AUDIO_INPUT_VOCAB_SIZE, adapter_rank),
                          nn.GELU(),
                          nn.Linear(adapter_rank, hidden, bias=False))
            for _ in range(NUM_CODEBOOKS)
        ])
        self.head_base = nn.Linear(hidden, AUDIO_OUTPUT_VOCAB_SIZE, bias=False)
        self.head_adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, adapter_rank, bias=False),
                          nn.GELU(),
                          nn.Linear(adapter_rank, AUDIO_OUTPUT_VOCAB_SIZE, bias=False))
            for _ in range(NUM_CODEBOOKS)
        ])
        self.semantic_projection = _projection(hidden, config.rms_norm_eps)
        self.codec_projection = _projection(hidden, config.rms_norm_eps)
        self.text_scale = nn.Parameter(torch.tensor(3.0))
        self.audio_scale = nn.Parameter(torch.tensor(1.0))
        # Match the Thinker at construction, including BF16/FP16 and device.
        reference = thinker_model.get_input_embeddings().weight
        self.to(device=reference.device, dtype=reference.dtype)
        if initialize:
            for target, source in zip(self.decoder.layers,
                                      thinker_model.layers[-num_layers:]):
                target.load_state_dict(source.state_dict(), strict=True)
            self.decoder.norm.load_state_dict(thinker_model.norm.state_dict(), strict=True)
        # Do not deepcopy layers: constructors assign independent cache indices.

    @property
    def heads(self):
        """Indexed, iterable callables; shared parameters are not re-registered."""
        return tuple(partial(self.project, codebook=q) for q in range(NUM_CODEBOOKS))

    def project(self, hidden, codebook):
        if isinstance(codebook, bool) or not isinstance(codebook, int) or not 0 <= codebook < NUM_CODEBOOKS:
            raise ValueError("codebook must be in 0..7")
        return self.head_base(hidden) + self.head_adapters[codebook](hidden)

    def embed_audio(self, audio_inputs):
        if (audio_inputs.ndim != 3 or audio_inputs.shape[-1] != NUM_CODEBOOKS
                or audio_inputs.dtype not in _INTEGER_DTYPES):
            raise ValueError("audio_inputs must be integer [batch, time, 8]")
        if torch.any((audio_inputs < 0) | (audio_inputs >= AUDIO_INPUT_VOCAB_SIZE)):
            raise ValueError("audio input IDs must be in 0..2111, including PAD and STOP")
        ids = audio_inputs
        streams = [self.embedding_base(ids[..., q]) + self.embedding_adapters[q](ids[..., q])
                   for q in range(NUM_CODEBOOKS)]
        return torch.stack(streams, dim=0).mean(dim=0)

    def forward(self, hidden):
        """Return [batch, time, 8, 2112] audio logits (or use heads individually)."""
        shared = self.head_base(hidden)
        return torch.stack([shared + adapter(hidden) for adapter in self.head_adapters], dim=-2)


class Qwen3ThinkerTalker(nn.Module):
    """Thinker with optional speech slots and differentiable semantic bridge.

    ``thinker`` may be Qwen3ForCausalLM or its Qwen3Model decoder. Returned
    caches are a pair of independent Transformers Cache instances. An attention
    mask during cached decoding covers the entire prefix plus current tokens.
    """

    def __init__(self, thinker, num_talker_layers=4, bridge_layer=None,
                 adapter_rank=256, initialize_from_thinker=True):
        super().__init__()
        core = thinker if isinstance(thinker, Qwen3Model) else getattr(thinker, "model", None)
        if not isinstance(core, Qwen3Model):
            raise ValueError("thinker must be Qwen3Model or Qwen3ForCausalLM")
        config = thinker.config
        _positive_int("num_hidden_layers", config.num_hidden_layers)
        _positive_int("num_talker_layers", num_talker_layers)
        _positive_int("adapter_rank", adapter_rank)
        _positive_int("hidden_size", config.hidden_size)
        _positive_int("num_attention_heads", config.num_attention_heads)
        _positive_int("num_key_value_heads", config.num_key_value_heads)
        _positive_int("head_dim", config.head_dim)
        _positive_int("max_position_embeddings", config.max_position_embeddings)
        if config.num_attention_heads % config.num_key_value_heads or config.head_dim % 2:
            raise ValueError("Qwen3 requires divisible query/KV heads and even rotary head_dim")
        if num_talker_layers > config.num_hidden_layers:
            raise ValueError("num_talker_layers cannot exceed Thinker depth")
        if len(core.layers) != config.num_hidden_layers:
            raise ValueError("Thinker config depth does not match its decoder")
        if bridge_layer is None:
            bridge_layer = max(0, config.num_hidden_layers // 2 - 1)
        if (isinstance(bridge_layer, bool) or not isinstance(bridge_layer, int)
                or not 0 <= bridge_layer < config.num_hidden_layers):
            raise ValueError("bridge_layer must index a Thinker block")
        if not isinstance(initialize_from_thinker, bool):
            raise ValueError("initialize_from_thinker must be boolean")
        self.thinker = thinker
        self.bridge_layer = bridge_layer
        self.adapter_rank = adapter_rank
        self.initialize_from_thinker = initialize_from_thinker
        self.audio_streams = TalkerModule(core, num_talker_layers, adapter_rank,
                                         initialize_from_thinker)

    @property
    def config(self):
        return self.thinker.config

    @property
    def _thinker_decoder(self):
        return self.thinker if isinstance(self.thinker, Qwen3Model) else self.thinker.model

    def get_input_embeddings(self):
        return self.thinker.get_input_embeddings()

    def talker_config(self):
        return dict(num_talker_layers=len(self.audio_streams.decoder.layers),
                    bridge_layer=self.bridge_layer, adapter_rank=self.adapter_rank)

    def forward_streams(self, input_ids, audio_inputs, attention_mask=None,
                        past_key_values=None, use_cache=False, speech_embeddings=None,
                        speech_mask=None):
        if (input_ids.ndim != 2 or input_ids.dtype not in _INTEGER_DTYPES
                or not all(input_ids.shape)):
            raise ValueError("input_ids must be nonempty integer [batch, time]")
        if torch.any((input_ids < 0) | (input_ids >= self.get_input_embeddings().num_embeddings)):
            raise ValueError("text input IDs are outside the Thinker vocabulary")
        if audio_inputs.shape != (*input_ids.shape, NUM_CODEBOOKS):
            raise ValueError("audio_inputs must match text batch/time and have 8 codebooks")
        if audio_inputs.device != input_ids.device:
            raise ValueError("text and audio inputs must be on the same device")
        thinker_cache = talker_cache = None
        cached_length = 0
        if past_key_values is not None:
            if not isinstance(past_key_values, tuple) or len(past_key_values) != 2:
                raise ValueError("past_key_values must be (thinker_cache, talker_cache)")
            thinker_cache, talker_cache = past_key_values
            if not use_cache:
                raise ValueError("past_key_values requires use_cache=True")
            if any(c is None or not hasattr(c, "get_seq_length") for c in past_key_values):
                raise ValueError("both caches must be Transformers Cache instances")
            if thinker_cache is talker_cache:
                raise ValueError("Thinker and Talker caches must be independent")
            cached_length = int(thinker_cache.get_seq_length())
            if cached_length != int(talker_cache.get_seq_length()):
                raise ValueError("Thinker and Talker cache lengths differ")
        total_length = cached_length + input_ids.shape[1]
        if total_length > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds configured context capacity")
        if attention_mask is not None:
            if attention_mask.shape != (input_ids.shape[0], total_length):
                raise ValueError("attention_mask must cover cached prefix and current tokens")
            if attention_mask.device != input_ids.device or torch.any((attention_mask != 0) & (attention_mask != 1)):
                raise ValueError("attention_mask must be binary and on the input device")
        if use_cache and self.training and (getattr(self._thinker_decoder, "gradient_checkpointing", False)
                                            or getattr(self.audio_streams.decoder, "gradient_checkpointing", False)):
            raise ValueError("use_cache is incompatible with training gradient checkpointing")
        thinker_inputs = {"input_ids": input_ids}
        if (speech_embeddings is None) != (speech_mask is None):
            raise ValueError("speech_embeddings and speech_mask must be supplied together")
        if speech_embeddings is not None:
            if (speech_embeddings.shape != (*input_ids.shape, self.config.hidden_size)
                    or speech_embeddings.device != input_ids.device
                    or speech_mask.shape != input_ids.shape or speech_mask.dtype != torch.bool
                    or speech_mask.device != input_ids.device or not speech_mask.any()
                    or not torch.isfinite(speech_embeddings).all()):
                raise ValueError("invalid speech embeddings or boolean slot mask")
            if cached_length or (attention_mask is not None and
                                 (speech_mask & ~attention_mask[:, -input_ids.shape[1]:].bool()).any()):
                raise ValueError("speech slots must be attended, uncached prefix positions")
            embeddings = self.get_input_embeddings()(input_ids)
            thinker_inputs = {"inputs_embeds": torch.where(
                speech_mask.unsqueeze(-1), speech_embeddings.to(embeddings.dtype), embeddings)}
        # Validate all audio inputs before either mutable KV cache is advanced.
        codec = self.audio_streams.embed_audio(audio_inputs)
        pre_norm = []
        handle = None
        if self.bridge_layer == self.config.num_hidden_layers - 1:
            # hidden_states[-1] is final-normalized, not the last block output.
            handle = self._thinker_decoder.norm.register_forward_pre_hook(
                lambda module, args: pre_norm.append(args[0]))
        try:
            text = self._thinker_decoder(
                **thinker_inputs, attention_mask=attention_mask,
                past_key_values=thinker_cache, use_cache=use_cache,
                output_hidden_states=True, return_dict=True,
            )
        finally:
            if handle is not None:
                handle.remove()
        semantic = (pre_norm[0] if pre_norm else text.hidden_states[self.bridge_layer + 1])
        audio_embeds = (self.audio_streams.text_scale * self.audio_streams.semantic_projection(semantic)
                        + self.audio_streams.audio_scale * self.audio_streams.codec_projection(codec))
        audio = self.audio_streams.decoder(
            inputs_embeds=audio_embeds, attention_mask=attention_mask,
            past_key_values=talker_cache, use_cache=use_cache, return_dict=True,
        )
        return SimpleNamespace(text_hidden=text.last_hidden_state,
                               audio_hidden=audio.last_hidden_state,
                               past_key_values=(text.past_key_values, audio.past_key_values)
                               if use_cache else None)
