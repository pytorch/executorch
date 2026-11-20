# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
    QnnExecuTorchHtpPrecision,
)
from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
    flatbuffer_to_option,
)
from executorch.backends.qualcomm.utils import utils


@pytest.mark.parametrize("backend", ["htp", "gpu", "lpai"])
def test_from_options_matches_explicit_compiler_specs(backend):
    if backend == "lpai" and utils.is_qnn_sdk_version_less_than("2.39"):
        with pytest.raises(ValueError, match="requires QNN SDK version >= 2.39"):
            QnnPartitioner.from_options("SM8850", backend=backend)
        return
    backend_options = {"use_fp16": True} if backend == "htp" else {}
    partitioner = QnnPartitioner.from_options(
        "SM8850",
        backend=backend,
        shared_buffer=True,
        skip_node_id_set={"add"},
        skip_mutable_buffer=True,
        **backend_options,
    )
    specs = utils.generate_qnn_executorch_compiler_spec(
        utils.get_soc_to_chipset_map()["SM8850"],
        getattr(utils, f"generate_{backend}_compiler_spec")(**backend_options),
        shared_buffer=True,
    )
    assert [(s.key, s.value) for s in partitioner.compiler_specs] == [
        (s.key, s.value) for s in specs
    ]
    assert partitioner.skip_node_id_set == {"add"}
    assert partitioner.skip_mutable_buffer


def test_from_options_routes_backend_and_partitioner_settings():
    partitioner = QnnPartitioner.from_options(
        "SM8750",
        use_fp16=True,
        shared_buffer=True,
        skip_node_op_set={"aten.add.Tensor"},
        convert_linear_to_conv2d=True,
    )
    options = flatbuffer_to_option(partitioner.compiler_specs[0].value)
    assert options.backend_options.backend_type == QnnExecuTorchBackendType.kHtpBackend
    assert (
        options.backend_options.htp_options.precision
        == QnnExecuTorchHtpPrecision.kHtpFp16
    )
    assert options.shared_buffer
    assert partitioner.convert_linear_to_conv2d
    assert partitioner.skip_node_op_set == {"aten.add.Tensor"}


@pytest.mark.parametrize("backend", ["htp", "gpu", "lpai"])
def test_from_options_rejects_unknown_option(backend):
    with pytest.raises(TypeError, match="Unknown QnnPartitioner option 'typo'"):
        QnnPartitioner.from_options("SM8750", backend=backend, typo=True)


def test_from_options_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported QNN backend"):
        QnnPartitioner.from_options("SM8750", backend="invalid")
