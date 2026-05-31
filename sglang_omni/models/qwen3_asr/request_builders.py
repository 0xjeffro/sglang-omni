# SPDX-License-Identifier: Apache-2.0
"""StagePayload <-> SGLang request adapters for Qwen3-ASR.

Unlike Whisper (encoder-decoder, features consumed inside the model forward),
Qwen3-ASR is a Qwen3 causal LM that ingests audio as multimodal embeddings:
the prompt contains an ``<|audio_pad|>`` placeholder repeated once per audio
token, and the model's ``general_mm_embed_routine`` scatters the encoder output
into those positions. So request_builder must:
  * extract mel features (WhisperFeatureExtractor) + attention mask,
  * compute how many audio tokens the encoder will emit,
  * build the chat prompt with that many ``<|audio_pad|>`` tokens,
  * hand the features over as a ``MultimodalDataItem``.
"""

from __future__ import annotations

import hashlib
import io
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers import GenerationConfig

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

_SAMPLE_RATE = 16000

_AUDIO_START = "<|audio_start|>"
_AUDIO_PAD = "<|audio_pad|>"
_AUDIO_END = "<|audio_end|>"

# Qwen3-ASR emits "language English<asr_text>...". Strip that lead-in so the
# returned text is just the transcription (mirrors upstream serving_transcription).
_ASR_TEXT_RE = re.compile(r"^.*?<asr_text>", re.DOTALL)


@dataclass
class Qwen3ASRRequestData(SGLangARRequestData):
    prompt_token_ids: list[int] | None = None
    output_ids: list[int] | None = None
    audio_duration_s: float = 0.0
    language: str = "en"
    engine_start_s: float = 0.0


def _audio_source_from_payload(payload: StagePayload) -> Any:
    inputs = payload.request.inputs
    if isinstance(inputs, dict):
        for key in ("audio_bytes", "bytes", "file"):
            value = inputs.get(key)
            if value is not None:
                return value
        for key in ("audio_path", "path", "url"):
            value = inputs.get(key)
            if value is not None:
                return value
    return inputs


def load_audio(source: Any) -> np.ndarray:
    import torchaudio

    if isinstance(source, memoryview):
        source = source.tobytes()
    if isinstance(source, bytearray):
        source = bytes(source)

    if isinstance(source, bytes):
        audio, sample_rate = torchaudio.load(io.BytesIO(source))
    elif isinstance(source, str):
        audio, sample_rate = torchaudio.load(source)
    else:
        raise ValueError(
            f"Unsupported Qwen3-ASR audio input: {type(source).__name__}"
        )

    if audio.ndim == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.squeeze(0).to(torch.float32)
    if sample_rate != _SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sample_rate, _SAMPLE_RATE)
    return audio.cpu().numpy()


def _audio_fingerprint(audio: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(audio, dtype=np.float32)
    return hashlib.blake2b(contiguous.tobytes(), digest_size=16).hexdigest()


def _audio_fingerprint_int(fingerprint: str) -> int:
    return int(fingerprint[:16], 16)


def _num_audio_tokens(num_mel_frames: int) -> int:
    """Audio tokens emitted by the Qwen3-ASR encoder for a mel of given length.

    Mirrors ``_get_feat_extract_output_lengths`` in the upstream modeling code
    (3 stride-2 conv downsamples + windowed packing of 100-frame windows).
    """
    leave = num_mel_frames % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (num_mel_frames // 100) * 13


def make_qwen3_asr_scheduler_adapters(
    *,
    processor: Any,
    tokenizer: Any,
    generation_config: GenerationConfig,
    encoder_token_count: int,
    max_new_tokens: int,
) -> tuple[
    Callable[[StagePayload], Qwen3ASRRequestData], Callable[[Any], StagePayload]
]:
    feature_extractor = getattr(processor, "feature_extractor", None)
    if feature_extractor is None:
        raise ValueError("Qwen3-ASR processor is missing a feature_extractor")

    eos_token_id = int(tokenizer.eos_token_id)
    pad_token_id = int(getattr(tokenizer, "pad_token_id", None) or eos_token_id)
    vocab_size = int(tokenizer.vocab_size)

    def _build_prompt_ids(num_audio_tokens: int) -> list[int]:
        prompt = (
            f"<|im_start|>user\n"
            f"{_AUDIO_START}{_AUDIO_PAD * num_audio_tokens}{_AUDIO_END}"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return tokenizer(prompt, add_special_tokens=False).input_ids

    def request_builder(payload: StagePayload) -> Qwen3ASRRequestData:
        params = payload.request.params or {}
        audio = load_audio(_audio_source_from_payload(payload))
        audio_duration_s = float(len(audio) / _SAMPLE_RATE)
        fingerprint = _audio_fingerprint(audio)

        extracted = feature_extractor(
            audio,
            sampling_rate=_SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        features = extracted.input_features
        feature_attention_mask = getattr(extracted, "attention_mask", None)
        if feature_attention_mask is not None:
            num_mel_frames = int(feature_attention_mask.sum().item())
        else:
            num_mel_frames = int(features.shape[-1])
        num_audio_tokens = int(_num_audio_tokens(num_mel_frames))

        input_ids = _build_prompt_ids(num_audio_tokens)

        mm_item_kwargs = dict(
            modality=Modality.AUDIO,
            hash=_audio_fingerprint_int(fingerprint),
            feature=features,
        )
        if feature_attention_mask is not None:
            mm_item_kwargs["feature_attention_mask"] = feature_attention_mask
        mm_inputs = MultimodalInputs(
            mm_items=[MultimodalDataItem(**mm_item_kwargs)],
            num_image_tokens=num_audio_tokens,
        )

        temperature = float(params.get("temperature") or 0.0)
        request_max_new_tokens = int(params.get("max_new_tokens") or max_new_tokens)
        sampling_params = SamplingParams(
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            stop_token_ids=[eos_token_id],
        )
        sampling_params.normalize(tokenizer=None)

        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=input_ids,
            sampling_params=sampling_params,
            vocab_size=vocab_size,
            extra_key=fingerprint,
        )
        req.multimodal_inputs = mm_inputs
        req._codec_suppress_tokens = None

        return Qwen3ASRRequestData(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            req=req,
            prompt_token_ids=input_ids,
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            audio_duration_s=audio_duration_s,
            language=str(params.get("language") or "en"),
            engine_start_s=time.perf_counter(),
            stage_payload=payload,
        )

    def result_adapter(data: Qwen3ASRRequestData) -> StagePayload:
        payload = data.stage_payload
        output_ids = list(data.output_ids or [])
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        text = _ASR_TEXT_RE.sub("", text, count=1).strip()
        engine_time_s = (
            time.perf_counter() - data.engine_start_s if data.engine_start_s else 0.0
        )
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "text": text,
                "language": data.language,
                "duration_s": data.audio_duration_s,
                "asr_latency_s": engine_time_s,
                "usage": {"engine_time_s": engine_time_s},
                "modality": "text",
            },
        )

    return request_builder, result_adapter


__all__ = [
    "Qwen3ASRRequestData",
    "load_audio",
    "make_qwen3_asr_scheduler_adapters",
]
