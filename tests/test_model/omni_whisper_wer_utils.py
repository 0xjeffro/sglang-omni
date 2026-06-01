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
ASR_WORKER_ARGS = "--stages.0.factory-args.max-running-requests 1"
ASR_ROUTER_STARTUP_TIMEOUT = 600
REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_CLEANUP_SCRIPT = REPO_ROOT / ".github/scripts/ensure_gpus_idle.sh"
# Colocated upstream router + ASR router back-to-back needs headroom on both cards.
GPU_IDLE_THRESHOLD_MB = 2048
GPU_IDLE_WAIT_SECONDS = 600
GPU_IDLE_POLL_SECONDS = 5


@pytest.fixture
def omni_whisper_wer_router(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[ManagedRouterHandle]:
    """Launch DP=2 Qwen3-ASR router for WER after upstream servers release GPU."""
    wait_for_gpu_memory_release()
    with launch_managed_router(
        tmp_path_factory=tmp_path_factory,
        model_path=ASR_MODEL_PATH,
        model_name=ASR_MODEL_PATH,
        worker_extra_args=ASR_WORKER_ARGS,
        wait_timeout=ASR_ROUTER_STARTUP_TIMEOUT,
        log_prefix="asr_wer_router_logs",
    ) as router:
        yield router
