# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from importlib.util import find_spec

import pytest

from executorch.backends.cortex_m.compile_config import CortexMCompileConfig

_HAS_CMSIS_NN = find_spec("cmsis_nn") is not None


class TestCortexMCompileConfig:
    def test_default_is_m55_mve(self):
        config = CortexMCompileConfig()
        assert config.cpu == "cortex-m55"
        assert config.isa == "mve"

    @pytest.mark.parametrize(
        "target_string,expected_cpu,expected_isa",
        [
            ("cortex-m0+int8", "cortex-m0", "scalar"),
            ("cortex-m0plus+int8", "cortex-m0plus", "scalar"),
            ("cortex-m3+int8", "cortex-m3", "scalar"),
            ("cortex-m4+int8", "cortex-m4", "dsp"),
            ("cortex-m7+int8", "cortex-m7", "dsp"),
            ("cortex-m23+int8", "cortex-m23", "scalar"),
            ("cortex-m33+int8", "cortex-m33", "dsp"),
            ("cortex-m35p+int8", "cortex-m35p", "dsp"),
            ("cortex-m52+int8", "cortex-m52", "mve"),
            ("cortex-m55+int8", "cortex-m55", "mve"),
            ("cortex-m85+int8", "cortex-m85", "mve"),
        ],
    )
    def test_from_target_string(self, target_string, expected_cpu, expected_isa):
        config = CortexMCompileConfig.from_target_string(target_string)
        assert config.cpu == expected_cpu
        assert config.isa == expected_isa

    def test_from_target_string_rejects_unknown_cpu(self):
        with pytest.raises(ValueError, match="cortex-m999"):
            CortexMCompileConfig.from_target_string("cortex-m999+int8")

    @pytest.mark.parametrize(
        "target_string",
        [
            "cortex-m55",  # missing feature suffix
            "cortex-m55+int8+int16",  # unsupported extra feature
            "cortex-m55+",  # trailing plus
            "cortex-m55+fp16",  # unknown feature
        ],
    )
    def test_from_target_string_rejects_invalid_features(self, target_string):
        with pytest.raises(ValueError):
            CortexMCompileConfig.from_target_string(target_string)

    def test_default_matches_m55_target_string(self):
        # Regression guard: pre-Phase-1 behavior was M55+MVE; the default
        # constructor must remain equivalent to parsing the existing target.
        assert CortexMCompileConfig() == CortexMCompileConfig.from_target_string(
            "cortex-m55+int8"
        )

    def test_is_hashable_and_frozen(self):
        from dataclasses import FrozenInstanceError

        config = CortexMCompileConfig(cpu="cortex-m33")
        assert hash(config) == hash(CortexMCompileConfig(cpu="cortex-m33"))
        assert {config, CortexMCompileConfig(cpu="cortex-m33")} == {config}
        with pytest.raises(FrozenInstanceError):
            config.cpu = "cortex-m55"  # type: ignore[misc]

    def test_explicit_isa_override(self):
        config = CortexMCompileConfig(cpu="cortex-m33", isa="scalar")
        assert config.cpu == "cortex-m33"
        assert config.isa == "scalar"


@pytest.mark.skipif(
    not _HAS_CMSIS_NN, reason="cortex_m passes require cmsis_nn"
)
class TestPassManagerConfigWiring:
    def test_default_config_is_m55(self):
        from executorch.backends.cortex_m.passes.cortex_m_pass_manager import (
            CortexMPassManager,
        )

        pm = CortexMPassManager(exported_program=None)
        assert pm.config.cpu == "cortex-m55"
        assert pm.config.isa == "mve"

    def test_explicit_config_threaded(self):
        from executorch.backends.cortex_m.passes.cortex_m_pass_manager import (
            CortexMPassManager,
        )

        config = CortexMCompileConfig(cpu="cortex-m33")
        pm = CortexMPassManager(exported_program=None, config=config)
        assert pm.config.cpu == "cortex-m33"
        assert pm.config.isa == "dsp"
