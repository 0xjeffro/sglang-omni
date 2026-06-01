# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib


def test_seedtts_tts_and_asr_concurrency_defaults_for_dp2(monkeypatch) -> None:
    monkeypatch.delenv("TTS_BENCHMARK_CONCURRENCY", raising=False)
    monkeypatch.delenv("QWEN3_ASR_CONCURRENCY", raising=False)
    monkeypatch.delenv("SEEDTTS_ASR_CONCURRENCY", raising=False)

    tts_tasks = importlib.import_module("benchmarks.tasks.tts")
    tts_seedtts = importlib.import_module("benchmarks.eval.benchmark_tts_seedtts")
    omni_seedtts = importlib.import_module("benchmarks.eval.benchmark_omni_seedtts")
    omni_mmmu = importlib.import_module("benchmarks.eval.benchmark_omni_mmmu")
    omni_videomme = importlib.import_module("benchmarks.eval.benchmark_omni_videomme")
    qwen3_asr_wer_utils = importlib.import_module(
        "tests.test_model.qwen3_asr_wer_utils"
    )
    tts_tasks = importlib.reload(tts_tasks)
    tts_seedtts = importlib.reload(tts_seedtts)
    omni_seedtts = importlib.reload(omni_seedtts)
    omni_mmmu = importlib.reload(omni_mmmu)
    omni_videomme = importlib.reload(omni_videomme)
    qwen3_asr_wer_utils = importlib.reload(qwen3_asr_wer_utils)

    tts_config = tts_seedtts.TtsSeedttsBenchmarkConfig(model="tts", meta="meta")
    omni_config = omni_seedtts.OmniSeedttsBenchmarkConfig(model="omni", meta="meta")
    mmmu_config = omni_mmmu.MMMUEvalConfig(model="omni")
    videomme_config = omni_videomme.VideoEvalConfig(model="omni")

    assert tts_config.concurrency == 16
    assert omni_config.max_concurrency == 16
    assert tts_config.asr_concurrency == 32
    assert omni_config.asr_concurrency == 32
    assert mmmu_config.asr_concurrency == 32
    assert videomme_config.asr_concurrency == 32
    assert tts_tasks.DEFAULT_ASR_TRANSCRIBE_CONCURRENCY == 32
    assert qwen3_asr_wer_utils.QWEN3_ASR_WER_CONCURRENCY == 32
