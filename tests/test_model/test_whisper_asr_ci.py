# SPDX-License-Identifier: Apache-2.0
"""Legacy workflow entrypoint for Qwen3-ASR correctness CI.

The real test implementation lives in ``test_qwen3_asr_ci.py``.  This wrapper
keeps the existing GitHub Actions command working until the workflow can be
updated with credentials that have the ``workflow`` scope.
"""

from tests.test_model.test_qwen3_asr_ci import *  # noqa: F401,F403
