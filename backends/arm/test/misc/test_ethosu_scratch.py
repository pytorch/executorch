# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from executorch.backends.arm import arm_vela
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.ethosu.backend import EthosUBackend
from executorch.backends.arm.scripts.aot_arm_compiler import _get_compile_spec
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import EthosUPipelineINTBase
from executorch.backends.arm.tosa.backend import TOSABackend
from executorch.exir.backend.backend_details import PreprocessResult


@pytest.fixture
def compile_scratch(tmp_path):
    def compile_with_scratch(scratch_size, compile_spec):
        def vela_main(args):
            output_dir = next(
                Path(arg.split("=", 1)[1])
                for arg in args
                if arg.startswith("--output-dir=")
            )
            output_dir.mkdir(exist_ok=True)
            np.savez(
                output_dir / "out_vela.npz",
                cmd_data=np.zeros(16, dtype=np.uint8),
                weight_data=np.zeros(16, dtype=np.uint8),
                scratch_shape=np.array([scratch_size], dtype=np.int64),
                input_shape=np.empty((0, 6), dtype=np.int32),
                output_shape=np.empty((0, 6), dtype=np.int32),
            )

        compile_spec.dump_intermediate_artifacts_to(str(tmp_path))
        with (
            patch.object(arm_vela, "has_vela", True),
            patch.object(
                arm_vela, "vela", SimpleNamespace(main=vela_main), create=True
            ),
            patch.object(
                TOSABackend,
                "_preprocess",
                return_value=PreprocessResult(processed_bytes=b"tosa"),
            ),
        ):
            return EthosUBackend.preprocess(None, compile_spec._to_list())

    return compile_with_scratch


@pytest.mark.parametrize("scratch_size", [0, 2097136, 2097152])
def test_u55_scratch_within_capacity(compile_scratch, scratch_size):
    result = compile_scratch(scratch_size, common.get_u55_compile_spec())
    assert result.processed_bytes.startswith(b"vela_bin_stream")


@pytest.mark.parametrize("config", [None, "Arm/vela.ini"])
@pytest.mark.parametrize(
    "scratch_size",
    [
        pytest.param(2097168, id="over_capacity"),
        pytest.param(2769536, id="reported_convnext_small"),
        pytest.param(2953632, id="reported_densenet161"),
        pytest.param(3637888, id="reported_maxvit_t"),
    ],
)
def test_u55_scratch_exceeds_capacity(compile_scratch, scratch_size, config):
    with pytest.raises(
        RuntimeError,
        match=f"requires {scratch_size} bytes, exceeding the configured capacity "
        f"of 2097152 bytes by {scratch_size - 2097152} bytes",
    ):
        compile_scratch(scratch_size, common.get_u55_compile_spec(config=config))


@pytest.mark.parametrize(
    "system_config,config",
    [
        ("Custom_U55", None),
        ("Custom_U55", "Arm/vela.ini"),
        ("Ethos_U55_High_End_Embedded", "custom_vela.ini"),
        ("Custom_U55", "custom_vela.ini"),
    ],
)
def test_custom_u55_test_flow_scratch_capacity(compile_scratch, system_config, config):
    compile_spec = common.get_u55_compile_spec(
        system_config=system_config, config=config
    )
    assert compile_spec.max_scratch_size is None
    assert compile_scratch(3637888, compile_spec).processed_bytes


@pytest.mark.parametrize("max_scratch_size", [None, 4194304])
def test_optional_scratch_capacity(compile_scratch, max_scratch_size):
    compile_spec = EthosUCompileSpec("ethos-u55-128", max_scratch_size=max_scratch_size)
    assert compile_scratch(3637888, compile_spec).processed_bytes


@pytest.mark.parametrize(
    "compile_spec_factory", [common.get_u65_compile_spec, common.get_u85_compile_spec]
)
@pytest.mark.parametrize("kwargs", [{}, {"memory_mode": "Shared_Sram"}])
def test_other_ethosu_test_flows_unchanged(
    compile_scratch, compile_spec_factory, kwargs
):
    compile_spec = compile_spec_factory(**kwargs)
    assert compile_spec.max_scratch_size is None
    assert compile_scratch(3637888, compile_spec).processed_bytes


@pytest.mark.parametrize("system_config", [None, "Ethos_U55_High_End_Embedded"])
@pytest.mark.parametrize(
    "target,memory_mode,limit,expected",
    [
        ("ethos-u55-128", None, None, 2097152),
        ("ethos-u55-128", "Shared_Sram", None, 2097152),
        ("ethos-u55-128", "Shared_Sram", 4194304, 4194304),
        ("ethos-u55-128", "Sram_Only", None, None),
        ("ethos-u65-256", None, None, None),
        ("ethos-u85-128", "Dedicated_Sram_384KB", None, None),
    ],
)
def test_aot_scratch_capacity(target, system_config, memory_mode, limit, expected):
    args = SimpleNamespace(
        target=target,
        system_config=system_config,
        memory_mode=memory_mode,
        max_scratch_size=limit,
        config="Arm/vela.ini",
        enable_debug_mode=None,
        direct_drive=False,
        intermediates=None,
    )
    assert _get_compile_spec(args).max_scratch_size == expected


@pytest.mark.parametrize(
    "system_config,config",
    [
        ("Custom_U55", None),
        ("Custom_U55", "Arm/vela.ini"),
        (None, "custom_vela.ini"),
        ("Ethos_U55_High_End_Embedded", "custom_vela.ini"),
        ("Custom_U55", "custom_vela.ini"),
    ],
)
@pytest.mark.parametrize("limit", [None, 4194304])
def test_aot_custom_u55_scratch_capacity(compile_scratch, system_config, config, limit):
    args = SimpleNamespace(
        target="ethos-u55-128",
        system_config=system_config,
        memory_mode="Shared_Sram",
        max_scratch_size=limit,
        config=config,
        enable_debug_mode=None,
        direct_drive=False,
        intermediates=None,
    )
    compile_spec = _get_compile_spec(args)
    assert compile_spec.max_scratch_size == limit
    assert compile_scratch(3637888, compile_spec).processed_bytes


@pytest.mark.skipif(not arm_vela.has_vela, reason="Vela is not installed")
def test_real_vela_scratch_exceeds_capacity():
    pipeline = EthosUPipelineINTBase(
        EthosUCompileSpec("ethos-u55-128", max_scratch_size=1),
        torch.nn.ReLU(),
        (torch.randn(1, 3, 8, 8),),
        aten_ops=[],
        exir_ops=[],
        run_on_fvp=False,
    )
    with pytest.raises(
        RuntimeError, match="exceeding the configured capacity of 1 bytes"
    ):
        pipeline.run()
