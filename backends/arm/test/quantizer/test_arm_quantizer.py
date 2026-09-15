# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    TOSAQuantizer,
    VgfQuantizer,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.vgf import VgfCompileSpec


@pytest.mark.parametrize(
    "quantizer_class, compile_spec",
    [
        (TOSAQuantizer, TosaSpecification.create_from_string("TOSA-1.0+INT")),
        (EthosUQuantizer, EthosUCompileSpec("ethos-u55-128")),
        (VgfQuantizer, VgfCompileSpec("TOSA-1.0+INT")),
    ],
)
def test_legacy_quantizer_reports_removal(quantizer_class, compile_spec):
    with pytest.raises(ValueError, match="removed in ExecuTorch 1.6"):
        quantizer_class(compile_spec, use_composable_quantizer=False)
