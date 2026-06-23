# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import cmsis_nn  # type: ignore[import-not-found, import-untyped]
import pytest

from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig


class TestCortexMTargetConfig:
    @pytest.mark.parametrize(
        "target_string,expected_cpu",
        [
            ("cortex-m0", CortexM.M0),
            ("cortex-m0plus", CortexM.M0PLUS),
            ("cortex-m3", CortexM.M3),
            ("cortex-m4", CortexM.M4),
            ("cortex-m7", CortexM.M7),
            ("cortex-m23", CortexM.M23),
            ("cortex-m33", CortexM.M33),
            ("cortex-m35p", CortexM.M35P),
            ("cortex-m55", CortexM.M55),
            ("cortex-m85", CortexM.M85),
        ],
    )
    def test_from_target_string(self, target_string, expected_cpu):
        config = CortexMTargetConfig.from_target_string(target_string)
        assert config.cpu == expected_cpu

    @pytest.mark.parametrize(
        "cpu,expected_backend",
        [
            (CortexM.M0, cmsis_nn.Backend.SCALAR),
            (CortexM.M4, cmsis_nn.Backend.DSP),
            (CortexM.M33, cmsis_nn.Backend.DSP),
            (CortexM.M55, cmsis_nn.Backend.MVE),
            (CortexM.M85, cmsis_nn.Backend.MVE),
        ],
    )
    def test_backend_resolved_via_cmsis_nn(self, cpu, expected_backend):
        assert CortexMTargetConfig(cpu=cpu).backend == expected_backend

    @pytest.mark.parametrize(
        "cpu,override",
        [
            (CortexM.M55, cmsis_nn.Backend.DSP),  # M55 with MVE disabled
            (CortexM.M55, cmsis_nn.Backend.SCALAR),  # M55 without DSP or MVE
            (CortexM.M85, cmsis_nn.Backend.DSP),
            (CortexM.M33, cmsis_nn.Backend.SCALAR),  # M33 without DSP option
            (CortexM.M4, cmsis_nn.Backend.SCALAR),  # M4 without DSP intrinsics
        ],
    )
    def test_isa_override_compatible(self, cpu, override):
        config = CortexMTargetConfig(cpu=cpu, isa=override)
        assert config.backend == override

    @pytest.mark.parametrize(
        "cpu,override",
        [
            (CortexM.M0, cmsis_nn.Backend.DSP),  # Armv6-M has no DSP
            (CortexM.M0, cmsis_nn.Backend.MVE),
            (CortexM.M3, cmsis_nn.Backend.DSP),  # Armv7-M has no DSP
            (CortexM.M4, cmsis_nn.Backend.MVE),  # Armv7E-M has no MVE
            (CortexM.M33, cmsis_nn.Backend.MVE),  # Armv8-M Mainline has no MVE
            (CortexM.M35P, cmsis_nn.Backend.MVE),
        ],
    )
    def test_isa_override_rejects_incompatible(self, cpu, override):
        with pytest.raises(ValueError, match="not supported"):
            CortexMTargetConfig(cpu=cpu, isa=override)

    @pytest.mark.parametrize(
        "target_string",
        [
            "cortex-m999",
            "cortex-m52",  # not yet in cmsis_nn.CortexM
            "cortex-m55+int8",  # legacy +int8 form no longer accepted
            "arm-m4",
        ],
    )
    def test_from_target_string_rejects_invalid(self, target_string):
        with pytest.raises(ValueError):
            CortexMTargetConfig.from_target_string(target_string)

    def test_is_hashable_and_frozen(self):
        from dataclasses import FrozenInstanceError

        config = CortexMTargetConfig(cpu=CortexM.M33)
        assert hash(config) == hash(CortexMTargetConfig(cpu=CortexM.M33))
        assert {config, CortexMTargetConfig(cpu=CortexM.M33)} == {config}
        with pytest.raises(FrozenInstanceError):
            config.cpu = CortexM.M55  # type: ignore[misc]


class TestPassManagerTargetConfigWiring:
    def test_default_target_config_is_m55(self):
        from executorch.backends.cortex_m.passes.cortex_m_pass_manager import (
            CortexMPassManager,
        )

        pm = CortexMPassManager(exported_program=None)
        assert pm.target_config.cpu == CortexM.M55
        assert pm.target_config.backend == cmsis_nn.Backend.MVE

    def test_explicit_target_config_threaded(self):
        from executorch.backends.cortex_m.passes.cortex_m_pass_manager import (
            CortexMPassManager,
        )

        target_config = CortexMTargetConfig(cpu=CortexM.M33)
        pm = CortexMPassManager(exported_program=None, target_config=target_config)
        assert pm.target_config.cpu == CortexM.M33
        assert pm.target_config.backend == cmsis_nn.Backend.DSP
