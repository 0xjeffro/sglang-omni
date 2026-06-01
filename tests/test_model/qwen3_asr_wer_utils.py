# SPDX-License-Identifier: Apache-2.0
"""Shared Qwen3-ASR router helpers for CI WER evaluation."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from benchmarks.tasks.tts import (
    DEFAULT_ASR_TRANSCRIBE_CONCURRENCY,
    QWEN3_ASR_MODEL_PATH,
)
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    launch_managed_router,
)
from tests.utils import wait_for_gpu_memory_release

QWEN3_ASR_WER_MODEL_PATH = QWEN3_ASR_MODEL_PATH
QWEN3_ASR_WER_CONCURRENCY = DEFAULT_ASR_TRANSCRIBE_CONCURRENCY
QWEN3_ASR_ROUTER_STARTUP_TIMEOUT = 600


@pytest.fixture
def qwen3_asr_wer_router(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[ManagedRouterHandle]:
    """Launch Qwen3-ASR router for WER after upstream servers release GPU."""
    wait_for_gpu_memory_release()
    with launch_managed_router(
        tmp_path_factory=tmp_path_factory,
        model_path=QWEN3_ASR_WER_MODEL_PATH,
        model_name=QWEN3_ASR_WER_MODEL_PATH,
        worker_extra_args="",
        wait_timeout=QWEN3_ASR_ROUTER_STARTUP_TIMEOUT,
        log_prefix="asr_wer_router_logs",
    ) as router:
        yield router
