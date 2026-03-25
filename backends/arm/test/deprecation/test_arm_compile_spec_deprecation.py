# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.exir.backend.compile_spec_schema import CompileSpec


def test_output_order_workaround_warns_on_set():
    tosa_spec = TosaSpecification.create_from_string("TOSA-1.1+FP")
    spec = TosaCompileSpec(tosa_spec)
    with pytest.warns(DeprecationWarning):
        spec._set_compile_specs(tosa_spec, [], output_order_workaround=True)
    assert spec.tosa_spec == tosa_spec
    assert spec.output_order_workaround is True


def test_from_list_with_output_order_workaround_warns():
    compile_specs = [
        CompileSpec("tosa_spec", b"TOSA-1.1+FP"),
        CompileSpec("output_format", b"tosa"),
        CompileSpec("ouput_reorder_workaround", b"true"),
    ]
    with pytest.warns(DeprecationWarning):
        spec = TosaCompileSpec.from_list(compile_specs)
    assert isinstance(spec, TosaCompileSpec)


def test_set_output_order_workaround_warns():
    spec = TosaCompileSpec("TOSA-1.1+FP")
    with pytest.warns(DeprecationWarning):
        spec.set_output_order_workaround(True)
    with pytest.warns(DeprecationWarning):
        value = spec.get_output_order_workaround()
    assert value is True


def test_get_output_order_workaround_warns():
    spec = TosaCompileSpec("TOSA-1.1+FP")
    with pytest.warns(DeprecationWarning):
        value = spec.get_output_order_workaround()
    assert value is False
