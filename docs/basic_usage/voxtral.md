# Voxtral TTS Model Usage

This guide uses [`mistralai/Voxtral-4B-TTS-2603`](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) — Voxtral TTS (a ~3.4B autoregressive backbone paired with a flow-matching acoustic transformer) — with SGLang-Omni and the OpenAI-compatible API. The pipeline is `preprocessing → generation → vocoder`; output audio is 24 kHz.

Unlike voice-cloning models such as [S2-Pro](./tts.md) and [Higgs](./higgs_tts.md), Voxtral does **not** condition on reference audio. It selects a speaker from a built-in **voice preset** via the `voice` field (defaults to `cheerful_female`). Available presets are loaded from the `voice_embedding/` directory inside the model checkpoint.

> **Note:** Voxtral TTS does not document Chinese support; ZH results are not a quality signal.

## Prerequisites

```bash
docker pull frankleeeee/sglang-omni:dev
docker run -it --shm-size 32g --gpus all frankleeeee/sglang-omni:dev /bin/zsh
```

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install -v .

hf download mistralai/Voxtral-4B-TTS-2603
```

## Launch the Server

```bash
sgl-omni serve \
  --model-path mistralai/Voxtral-4B-TTS-2603 \
  --config examples/configs/voxtral_tts.yaml \
  --port 8000
```

## Use Curl

### Basic TTS (voice preset)

Voxtral picks a server-side speaker from `voice`; there is no reference-audio cloning.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "voice": "cheerful_female",
    "max_new_tokens": 4096
  }' \
  --output output.wav
```

### Streaming

Set `"stream": true` to receive base64-encoded WAV chunks over Server-Sent Events:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "voice": "cheerful_female",
    "stream": true
  }'
```

## Use Python

### Basic TTS

```python
import requests

SPEECH_INPUT = "Get the trust fund to the bank early."

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": SPEECH_INPUT,
        "voice": "cheerful_female",
        "max_new_tokens": 4096,
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### Streaming

Consume the SSE stream and reassemble the base64-encoded WAV chunks into one file:

```python
import base64, io, json, wave

import requests

payload = {
    "input": "Get the trust fund to the bank early.",
    "voice": "cheerful_female",
    "max_new_tokens": 4096,
    "stream": True,
    "response_format": "wav",
}

chunks = []
fmt = None
with requests.post(
    "http://localhost:8000/v1/audio/speech",
    json=payload,
    stream=True,
    timeout=600,
) as stream:
    stream.raise_for_status()
    for line in stream.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[len("data:"):].lstrip()
        if data == "[DONE]":
            break
        b64 = (json.loads(data).get("audio") or {}).get("data")
        if not b64:
            continue
        with wave.open(io.BytesIO(base64.b64decode(b64)), "rb") as w:
            if fmt is None:
                fmt = w.getnchannels(), w.getsampwidth(), w.getframerate()
            chunks.append(w.readframes(w.getnframes()))

assert fmt
nc, sw, fr = fmt
with wave.open("output_stream.wav", "wb") as w:
    w.setnchannels(nc)
    w.setsampwidth(sw)
    w.setframerate(fr)
    w.writeframes(b"".join(chunks))
```

## Request Parameters

The `/v1/audio/speech` endpoint shares one request schema across all TTS models
(`CreateSpeechRequest` in `sglang_omni/serve/protocol.py`), but Voxtral's
preprocessing only consumes the fields below. Other accepted fields —
`temperature`, `top_p`, `top_k`, `seed`, `references` / `ref_audio` / `ref_text`,
`instructions`, `language` — are **ignored** by Voxtral: it is not a
voice-cloning model, and its acoustic generation does not take sampling
parameters.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input` | string | (required) | Text to synthesize |
| `voice` | string | `"cheerful_female"` | Built-in speaker preset. `"default"` / empty / omitted falls back to `cheerful_female`. Loaded from the checkpoint's `voice_embedding/`. |
| `response_format` | string | `"wav"` | Output audio format |
| `stream` | bool | `false` | Enable streaming via SSE |
| `max_new_tokens` | int | `4096` | Maximum number of generated frames |

## Benchmark Results

On seed-tts EN (full set, 1088 utterances), bf16, `max_new_tokens=4096`,
`--no-ref-audio --voice cheerful_female`, c=16, scored with HF Whisper-large-v3
for WER.

Hardware: 1 x H100 SXM

| metric | value |
|---|---|
| WER (corpus, micro-avg) | 1.15% |
| WER per-sample mean | 1.14% |
| WER per-sample median | 0.00% |
| WER per-sample std | 3.66% |
| WER per-sample p95 | 9.09% |
| WER per-sample max | 42.86% |
| >50% WER samples | 0 (0.0%) |
| Evaluated / Skipped | 1088/1088 / 0 |
| Concurrency | 16 |
| Completed requests | 1088 |
| Failed requests | 0 |
| Latency mean (s) | 3.321 |
| Latency median (s) | 3.252 |
| Latency p95 (s) | 5.003 |
| Latency p99 (s) | 6.17 |
| RTF mean | 0.5834 |
| RTF median | 0.5761 |
| Audio duration mean (s) | 5.711 |
| Output throughput (tok/s) | 341.5 |
| Output tokens (mean) | 71 |
| Output tokens (total) | 77675 |
| Prompt tokens (mean) | 150 |
| Prompt tokens (total) | 163577 |
| Throughput (req/s) | 4.784 |
