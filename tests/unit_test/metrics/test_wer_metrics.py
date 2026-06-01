# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from benchmarks.metrics.wer import calculate_asr_speed_metrics


def test_asr_speed_metrics_use_wall_time_when_available() -> None:
    outputs = [
        SimpleNamespace(is_success=True, asr_latency_s=0.4, audio_duration_s=2.0),
        SimpleNamespace(is_success=True, asr_latency_s=0.6, audio_duration_s=3.0),
    ]

    metrics = calculate_asr_speed_metrics(outputs, wall_time_s=0.7)

    assert metrics["asr_total_time_s"] == 0.7
    assert metrics["asr_latency_sum_s"] == 1.0
    assert metrics["asr_throughput_samples_per_s"] == 2 / 0.7
