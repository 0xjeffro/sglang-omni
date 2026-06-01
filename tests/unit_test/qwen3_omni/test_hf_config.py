# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang_omni.models.qwen3_omni.hf_config import Qwen3OmniMoeTextConfig


def test_qwen3_omni_mrope_scaling_uses_sglang_default_rope_type() -> None:
    config = Qwen3OmniMoeTextConfig(
        rope_scaling={
            "type": "mrope",
            "mrope_section": [16, 24, 24],
        }
    )

    assert config.rope_scaling is not None
    assert config.rope_scaling["type"] == "default"
    assert config.rope_scaling["rope_type"] == "default"
    assert config.rope_scaling["mrope_section"] == [16, 24, 24]
