# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings

from executorch.backends.arm.common.pipeline_config import SoftmaxDecompositionConfig
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.vgf import VgfCompileSpec
from pytest import raises, warns


def test_compile_spec_u55_INT():
    compile_spec = (
        EthosUCompileSpec("ethos-u55", extra_flags=["--my-flag"])
        .dump_intermediate_artifacts_to("my_path")
        .dump_debug_info(EthosUCompileSpec.DebugMode.TOSA)
    )
    spec_list = compile_spec._to_list()

    assert EthosUCompileSpec._from_list(spec_list) == compile_spec
    assert "--my-flag" in compile_spec.compiler_flags
    assert "--output-format=raw" in compile_spec.compiler_flags
    with raises(ValueError, match="Incorrect output format"):
        VgfCompileSpec._from_list(spec_list)

    spec_list.pop(0)
    with raises(ValueError, match="No tosa_spec in compile spec."):
        EthosUCompileSpec._from_list(spec_list)


def test_ethos_u55_defaults_to_stable_softmax_u55_INT():
    """Test that EthosUCompileSpec for U55 defaults to STABLE softmax config."""
    compile_spec = EthosUCompileSpec("ethos-u55-128")
    pipeline_config = compile_spec._get_pass_pipeline_config()
    assert pipeline_config.softmax == SoftmaxDecompositionConfig.STABLE


def test_ethos_u85_defaults_to_masked_softmax_u85_INT():
    """Test that EthosUCompileSpec for U85 defaults to MASKED softmax config."""
    compile_spec = EthosUCompileSpec("ethos-u85-256")
    pipeline_config = compile_spec._get_pass_pipeline_config()
    assert pipeline_config.softmax == SoftmaxDecompositionConfig.MASKED


def test_compile_spec_vgf_no_quant():
    compile_spec = (
        VgfCompileSpec(compiler_flags=["--my-flag"])
        .dump_intermediate_artifacts_to("my_path")
        .dump_debug_info(None)
    )
    compile_spec2 = VgfCompileSpec(
        compiler_flags=["--my-flag2"]
    ).dump_intermediate_artifacts_to("my_path")

    spec_list = compile_spec._to_list()

    assert VgfCompileSpec._from_list(spec_list) == compile_spec
    assert VgfCompileSpec._from_list(spec_list) != compile_spec2
    with raises(ValueError, match="Incorrect output format"):
        EthosUCompileSpec._from_list(spec_list)


def test_compile_spec_vgf_uses_default_pipeline_config():
    compile_spec = VgfCompileSpec()
    pipeline_config = compile_spec._get_pass_pipeline_config()

    assert pipeline_config.is_default()


def test_compile_spec_tosa_INT():
    compile_spec = TosaCompileSpec("TOSA-1.0+INT")
    spec_list = compile_spec._to_list()

    assert TosaCompileSpec._from_list(spec_list) == compile_spec
    with raises(ValueError, match="Incorrect output format"):
        VgfCompileSpec._from_list(spec_list)


def test_preserve_io_quantization_roundtrip_vgf_FP_INT():
    compile_spec = VgfCompileSpec()._set_preserve_io_quantization(True)
    roundtripped = VgfCompileSpec._from_list(compile_spec._to_list())
    assert roundtripped.preserve_io_quantization is True


def test_preserve_io_quantization_warns_for_u55_INT():
    with warns(
        UserWarning,
        match="preserve_io_quantization=True is redundant for INT-only TOSA",
    ):
        EthosUCompileSpec("ethos-u55-128")._set_preserve_io_quantization(True)


def test_preserve_io_quantization_no_warn_for_vgf_FP_INT():
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        VgfCompileSpec()._set_preserve_io_quantization(True)
    assert len(recorded_warnings) == 0
