# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from benchmarks.metrics.wer import (
    calculate_asr_speed_metrics,
    print_asr_speed_summary,
    print_asr_wer_summary,
    print_wer_summary,
)


def test_asr_speed_metrics_use_wall_time_when_available() -> None:
    outputs = [
        SimpleNamespace(is_success=True, asr_latency_s=0.4, audio_duration_s=2.0),
        SimpleNamespace(is_success=True, asr_latency_s=0.6, audio_duration_s=3.0),
    ]

    metrics = calculate_asr_speed_metrics(outputs, wall_time_s=0.7)

    assert metrics["asr_total_time_s"] == 0.7
    assert metrics["asr_latency_sum_s"] == 1.0
    assert metrics["asr_throughput_samples_per_s"] == 2 / 0.7


def test_tts_wer_summary_labels_tts_model(capsys) -> None:
    print_wer_summary(
        {
            "lang": "en",
            "evaluated": 1,
            "total_samples": 1,
            "skipped": 0,
            "wer_corpus": 0.0,
        },
        "tts-model",
        "streaming",
    )

    captured = capsys.readouterr().out

    assert "TTS model:" in captured
    assert "tts-model" in captured
    assert "ASR model:" not in captured


def test_asr_speed_summary_labels_asr_model(capsys) -> None:
    print_asr_speed_summary(
        {
            "evaluated": 1,
            "total_samples": 1,
            "skipped": 0,
            "asr_concurrency": 32,
            "asr_rtf_p95": 0.1234,
        },
        "Qwen/Qwen3-ASR-1.7B",
    )

    captured = capsys.readouterr().out

    assert "ASR model:" in captured
    assert "Qwen/Qwen3-ASR-1.7B" in captured
    assert "ASR concurrency:" in captured
    assert "32" in captured
    assert "ASR RTF p95:" in captured


def test_asr_wer_summary_labels_asr_model(capsys) -> None:
    print_asr_wer_summary(
        {
            "lang": "en",
            "evaluated": 1,
            "total_samples": 1,
            "skipped": 0,
            "corpus_wer": 0.01,
            "wer_per_sample_max": 0.02,
        },
        "Qwen/Qwen3-ASR-1.7B",
    )

    captured = capsys.readouterr().out

    assert "ASR WER Benchmark Result" in captured
    assert "ASR model:" in captured
    assert "Qwen/Qwen3-ASR-1.7B" in captured
    assert "TTS model:" not in captured
