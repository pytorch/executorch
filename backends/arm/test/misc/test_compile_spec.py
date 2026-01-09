# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.vgf import VgfCompileSpec
from pytest import raises


def test_ethos_u_compile_spec_no_target():
    compile_spec = (
        EthosUCompileSpec("ethos-u55", extra_flags=["--my-flag"])
        .dump_intermediate_artifacts_to("my_path")
        .dump_debug_info(EthosUCompileSpec.DebugMode.TOSA)
    )
    spec_list = compile_spec.to_list()

    assert EthosUCompileSpec.from_list(spec_list) == compile_spec
    assert "--my-flag" in compile_spec.compiler_flags
    assert "--output-format=raw" in compile_spec.compiler_flags
    with raises(ValueError, match="Incorrect output format"):
        VgfCompileSpec.from_list(spec_list)

    spec_list.pop(0)
    with raises(ValueError, match="No tosa_spec in compile spec."):
        EthosUCompileSpec.from_list(spec_list)


def test_vgf_compile_spec_no_target():
    compile_spec = (
        VgfCompileSpec(compiler_flags=["--my-flag"])
        .dump_intermediate_artifacts_to("my_path")
        .dump_debug_info(None)
    )
    compile_spec2 = VgfCompileSpec(
        compiler_flags=["--my-flag2"]
    ).dump_intermediate_artifacts_to("my_path")

    spec_list = compile_spec.to_list()

    assert VgfCompileSpec.from_list(spec_list) == compile_spec
    assert VgfCompileSpec.from_list(spec_list) != compile_spec2
    with raises(ValueError, match="Incorrect output format"):
        EthosUCompileSpec.from_list(spec_list)


def test_tosa_compile_spec_no_target():
    compile_spec = TosaCompileSpec("TOSA-1.0+INT")
    spec_list = compile_spec.to_list()

    assert TosaCompileSpec.from_list(spec_list) == compile_spec
    with raises(ValueError, match="Incorrect output format"):
        VgfCompileSpec.from_list(spec_list)
