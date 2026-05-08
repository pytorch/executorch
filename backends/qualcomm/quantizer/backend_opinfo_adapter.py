# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file adapts backend_opinfo from the QNN SDK for use with ExecuTorch.
# The backend_opinfo is utilized to verify the quantization constraints for each operator.
import logging

import os
import sys
from typing import Any, List, Optional

from executorch.backends.qualcomm.quantizer.validators import (
    NormalizedConstraints,
    PortDatatypeConstraints,
    TensorQuantConstraint,
)
from executorch.backends.qualcomm.serialization.qc_schema import (
    QcomChipset,
    QnnExecuTorchBackendType,
)


def add_qnn_python_path():
    qnn_root = os.environ.get("QNN_SDK_ROOT")
    if not qnn_root:
        raise EnvironmentError("QNN_SDK_ROOT is not set")

    qnn_py = os.path.join(qnn_root, "lib", "python")
    if not os.path.isdir(qnn_py):
        raise FileNotFoundError(f"Not found: {qnn_py}")

    if qnn_py not in sys.path:
        sys.path.insert(0, qnn_py)


try:
    add_qnn_python_path()
    from qti.aisw.converters.common import backend_opinfo

    _HAS_BACKEND_OPINFO = True
except Exception:
    backend_opinfo = None
    _HAS_BACKEND_OPINFO = False


class _NoOpBackendOpInfo:
    def __init__(self, *args, **kwargs):
        pass

    def get_all_supported_ops(self) -> List[str]:
        return []

    def get_op_info(self, op_name: str):
        return []


class _NoOpNamespace:
    HTP = 1
    LPAI = 3
    BackendOpInfo = _NoOpBackendOpInfo


if not _HAS_BACKEND_OPINFO:
    logging.warning(
        "The backend_opinfo module couldn't be imported, so the abstract implementation will be used instead. This might be because $QNN_SDK_ROOT/lib/python isn't included in your PYTHONPATH, or the `BackendOpInfo` API isn't available in your QNN SDK version. Note that the `BackendOpInfo` API is supported starting from QNN SDK 2.41 and above."
    )
    backend_opinfo = _NoOpNamespace()


def get_backend_opinfo(backend: str, soc_model: QcomChipset):
    backend_type = getattr(backend_opinfo, backend.upper())
    # For qnn 2.41, it only supports HTP backend
    # It will support LPAI backend as soon as possible.
    if backend == str(QnnExecuTorchBackendType.kLpaiBackend):
        return _NoOpBackendOpInfo()
    try:
        return backend_opinfo.BackendOpInfo(backend_type, soc_model)
    except Exception:
        print(
            f"The 'BackendOpInfo' APIs may not be available for this backend {backend}."
        )
        return _NoOpBackendOpInfo()


# Helper functions for normalizing OpInfo objects (moved from backend_opinfo_adapter)
def _normalize_datatype_info(datatype_info: Any) -> PortDatatypeConstraints:
    """
    Convert DatatypeInfo → PortDatatypeConstraints (flatten QuantConstraintInfo list).
    """
    dtype = datatype_info.get_datatype()
    qcs = []

    for qc in datatype_info.get_quant_constraint_info():
        qcs.append(
            TensorQuantConstraint(
                encoding_types=list(qc.get_encoding_types()),
                is_symmetric=qc.is_symmetric(),
                axis=list(qc.get_axis()) if qc.get_axis() is not None else None,
                is_math_invariant=qc.is_math_invariant(),
                scale=qc.get_scale(),
                offset=qc.get_offset(),
            )
        )
    return PortDatatypeConstraints(
        dtype=dtype,
        constraints=qcs,
        applicable_from_current_dtype_onward=bool(
            datatype_info.is_applicable_from_current_dtype_onward()
        ),
    )


def _normalize_op_info(op_info: Any) -> NormalizedConstraints:
    """
    Convert OpInfo → NormalizedConstraints.
    """
    inputs = [_normalize_datatype_info(di) for di in op_info.get_input_info()]
    outputs = [_normalize_datatype_info(di) for di in op_info.get_output_info()]
    return NormalizedConstraints(inputs=inputs, outputs=outputs)


def constraints_loader(
    backend_opinfo, qnn_op: str
) -> Optional[List[NormalizedConstraints]]:
    """
    Adapter used by rules:
      1) Fetch all OpInfo variants via backendOpInfo.get_op_info(qnn_op).
      2) Normalize into NormalizedConstraints for validation.
    """
    op_infos = backend_opinfo.get_op_info(qnn_op)
    normalized_constraints_list = []
    for op_info in op_infos:
        normalized_constraints_list.append(_normalize_op_info(op_info))
    return normalized_constraints_list
