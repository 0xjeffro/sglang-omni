# SPDX-License-Identifier: Apache-2.0
"""Shared Omni ASR router helpers for CI WER evaluation."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from benchmarks.tasks.tts import QWEN3_ASR_MODEL_PATH
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    launch_managed_router,
)
from tests.utils import wait_for_gpu_memory_release

ASR_MODEL_PATH = QWEN3_ASR_MODEL_PATH
ASR_ROUTER_STARTUP_TIMEOUT = 600


@pytest.fixture
def omni_whisper_wer_router(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[ManagedRouterHandle]:
    """Launch Qwen3-ASR router for WER after upstream servers release GPU."""
    wait_for_gpu_memory_release()
    with launch_managed_router(
        tmp_path_factory=tmp_path_factory,
        model_path=ASR_MODEL_PATH,
        model_name=ASR_MODEL_PATH,
        worker_extra_args="",
        wait_timeout=ASR_ROUTER_STARTUP_TIMEOUT,
        log_prefix="asr_wer_router_logs",
    ) as router:
        yield router
