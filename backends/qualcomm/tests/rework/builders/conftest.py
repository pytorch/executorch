# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import numpy as np
import pytest
import torch

from executorch.backends.qualcomm.builders.qnn_constants import (
    OpElementWiseMultiply,
    OpElementWiseSin,
    OpTranspose,
    QNN_OP_PACKAGE_NAME_QTI_AISW,
)
from executorch.backends.qualcomm.serialization.qc_schema import (
    QcomChipset,
    QnnExecuTorchBackendType,
    QnnExecuTorchOpPackageInfo,
    QnnExecuTorchOpPackageOptions,
)
from executorch.backends.qualcomm.tests.rework.conftest import default_property
from executorch.backends.qualcomm.tests.rework.passes import conftest as pass_fixtures
from executorch.backends.qualcomm.tests.rework.src.builders_helper import (
    BuilderPipeline,
)
from executorch.backends.qualcomm.utils.qnn_manager_lifecycle import QnnManagerContext
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
)


compile_specs = pass_fixtures.compile_specs
pass_pipeline = pass_fixtures.pass_pipeline


@pytest.fixture
def qnn_compile_spec(compile_specs):
    specs = compile_specs(QnnExecuTorchBackendType.kHtpBackend)
    if specs is None:
        pytest.skip("HTP compile specs are unavailable in this QNN SDK")
    with QnnManagerContext({"forward": specs}):
        yield specs


@pytest.fixture(params=[default_property().soc_model, "SM8350"])
def runtime_layout_compile_spec(request, custom_op_compile_specs):
    soc_model = QcomChipset[request.param]
    specs = custom_op_compile_specs(soc_model)
    with QnnManagerContext({"forward": specs}):
        yield soc_model, specs


@pytest.fixture
def custom_op_compile_spec(custom_op_compile_specs, qnn_config):
    soc_name = (
        default_property().soc_model
        if qnn_config.enable_x86_64
        else qnn_config.soc_model
    )
    specs = custom_op_compile_specs(QcomChipset[soc_name])
    with QnnManagerContext({"forward": specs}):
        yield specs


@pytest.fixture
def custom_op_compile_specs(builder_pipeline):
    # Use built-in QNN operations to avoid an external test op package.
    infos = [
        info
        for info in builder_pipeline.op_package_infos
        if info.custom_op_name
        in {"qnn_builder_test.sin.default", "qnn_builder_test.multiply.default"}
    ]

    def build(soc_model):
        return generate_qnn_executorch_compiler_spec(
            soc_model=soc_model,
            backend_options=generate_htp_compiler_spec(use_fp16=True),
            op_package_options=QnnExecuTorchOpPackageOptions(op_package_infos=infos),
        )

    return build


@pytest.fixture(scope="module")
def builder_op_package_infos():
    library = torch.library.Library("qnn_builder_test", "FRAGMENT")

    def identity(input):
        return input.clone()

    def pair(input):
        return input.clone(), input.clone()

    def pair_inputs(first, second):
        return first.clone(), second.clone()

    def dequant(weight, scale):
        return (weight * scale.reshape(-1, 1, 1, 1)).to(torch.float16)

    infos = []
    for name, schema, implementation, qnn_op_type in (
        ("foo", "(Tensor input) -> Tensor", identity, "Foo"),
        ("bar", "(Tensor input) -> Tensor", identity, "Bar"),
        ("pair", "(Tensor input) -> (Tensor, Tensor)", pair, "Pair"),
        (
            "pair_inputs",
            "(Tensor first, Tensor second) -> (Tensor, Tensor)",
            pair_inputs,
            "Pair_Inputs",
        ),
        ("dequant", "(Tensor weight, Tensor scale) -> Tensor", dequant, "Dequant"),
        ("sin", "(Tensor input) -> Tensor", torch.sin, OpElementWiseSin.op_name),
        (
            "multiply",
            "(Tensor weight, Tensor scale) -> Tensor",
            torch.mul,
            OpElementWiseMultiply.op_name,
        ),
    ):
        library.define(f"{name}{schema}")
        library.impl(name, implementation, "CompositeExplicitAutograd")
        torch.library.register_fake(
            f"qnn_builder_test::{name}", implementation, lib=library
        )

        infos.append(
            QnnExecuTorchOpPackageInfo(
                custom_op_name=f"qnn_builder_test.{name}.default",
                op_package_name=(
                    QNN_OP_PACKAGE_NAME_QTI_AISW
                    if name in {"sin", "multiply"}
                    else "TestOpPackage"
                ),
                qnn_op_type_name=qnn_op_type,
            )
        )
    try:
        yield infos
    finally:
        library._destroy()


@pytest.fixture
def builder_pipeline(builder_op_package_infos, monkeypatch):
    pipeline = BuilderPipeline(builder_op_package_infos)
    add_tensor_param = PyQnnManager.PyQnnOpWrapper.AddTensorParam

    def record_tensor_param(operation, name, data_type, rank, dims, data, copy_data):
        result = add_tensor_param(
            operation, name, data_type, rank, dims, data, copy_data
        )
        if (
            operation.GetOpWrapper().GetOpType() == OpTranspose.op_name
            and name == OpTranspose.param_perm
        ):
            pipeline.transpose_params[operation] = (
                data_type,
                rank,
                tuple(dims),
                np.array(data, copy=True),
            )
        return result

    monkeypatch.setattr(
        PyQnnManager.PyQnnOpWrapper, "AddTensorParam", record_tensor_param
    )
    return pipeline
