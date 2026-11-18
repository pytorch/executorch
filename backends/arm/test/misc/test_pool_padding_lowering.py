# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import torch
from executorch.backends.arm.common.pipeline_config import ArmPassPipelineConfig
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.ethosu.backend import EthosUBackend
from executorch.backends.arm.tosa.backend import TOSABackend
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.exir import to_edge
from tosa.Op import Op  # type: ignore[import-untyped]
from tosa.TosaGraph import TosaGraph  # type: ignore[import-untyped]


class Pool(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.max_pool2d(x, 5, stride=5, padding=1)


class ConditionalPool(torch.nn.Module):
    def forward(self, pred, x):
        return torch.cond(pred, Pool(), Pool(), (x,))


@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("enabled", [None, False, True])
@pytest.mark.parametrize("dtype", [torch.int8, torch.int16])
def test_internal_pool_padding_option(enabled, conditional, dtype):
    inputs = (torch.ones(1, 8, 20, 20, dtype=dtype),)
    model = ConditionalPool() if conditional else Pool()
    if conditional:
        inputs = (torch.tensor(True), *inputs)
    program = to_edge(torch.export.export(model, inputs)).exported_program()
    compile_spec = TosaCompileSpec("TOSA-1.0+INT+int16+cf")
    if enabled is None:
        result = TOSABackend.preprocess(program, compile_spec._to_list())
    else:
        compile_spec.set_pass_pipeline_config(
            ArmPassPipelineConfig(large_stride_pool_padding=enabled)
        )
        result = TOSABackend._preprocess(program, compile_spec)
    graph = TosaGraph.GetRootAs(result.processed_bytes, 0)
    pad_count = 0
    pool_count = 0
    for r in range(graph.RegionsLength()):
        region = graph.Regions(r)
        for b in range(region.BlocksLength()):
            block = region.Blocks(b)
            for i in range(block.OperatorsLength()):
                op = block.Operators(i).Op()
                pad_count += op == Op.PAD
                pool_count += op == Op.MAX_POOL2D
    assert pool_count == (2 if conditional else 1)
    assert pad_count == (pool_count if enabled else 0)


@pytest.mark.parametrize("target", ["ethos-u55-128", "ethos-u65-256", "ethos-u85-128"])
def test_ethosu_filters_tosa_spec_before_lowering(target):
    program = torch.export.export(Pool(), (torch.ones(1, 8, 20, 20),))
    compile_spec = EthosUCompileSpec(target)

    # Stop before lowering: this test covers the Ethos-U/TOSA boundary.
    class LoweringReached(Exception):
        pass

    def check_spec(edge_program, tosa_spec):
        assert isinstance(tosa_spec, TosaCompileSpec)
        assert tosa_spec.compiler_flags == []
        assert tosa_spec._get_pass_pipeline_config().large_stride_pool_padding == (
            target == "ethos-u85-128"
        )
        raise LoweringReached

    with patch.object(TOSABackend, "_preprocess", side_effect=check_spec):
        with pytest.raises(LoweringReached):
            EthosUBackend.preprocess(program, compile_spec._to_list())


@pytest.mark.parametrize("enabled", [False, True])
def test_pool_padding_config_round_trip(enabled):
    config = ArmPassPipelineConfig(large_stride_pool_padding=enabled)
    assert config.is_default() == (not enabled)
    restored = ArmPassPipelineConfig.from_dict(config.to_dict())
    assert restored == config
