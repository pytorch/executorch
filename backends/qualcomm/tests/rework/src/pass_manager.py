# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.qualcomm._passes import (
    LpaiPartitionFallbackSupport,
    ResolveDebugHandle,
    TagQuantIO,
)
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_qnn_pass_manager_cls,
)
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.utils.constants import (
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
)
from executorch.exir.pass_base import ExportPass, PassResult


class _CustomPass(ExportPass):
    def call(self, graph_module):
        return PassResult(graph_module, False)


class PassManager:
    @staticmethod
    def test(backend_type: QnnExecuTorchBackendType):
        pass_manager_cls = get_qnn_pass_manager_cls(backend_type)
        terminal_passes = (
            (ResolveDebugHandle, LpaiPartitionFallbackSupport)
            if backend_type == QnnExecuTorchBackendType.kLpaiBackend
            else (ResolveDebugHandle,)
        )

        passes_job = pass_manager_cls.get_capture_program_passes()
        passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
        passes_job[_CustomPass] = {
            QCOM_PASS_ACTIVATE_KEY: True,
            QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY: {},
        }
        dep_table = pass_manager_cls.get_passes_dependency_for_capture_program()
        dep_table[_CustomPass] = [TagQuantIO]

        passes = pass_manager_cls().get_to_edge_transform_passes(
            exported_program=None,
            passes_job=passes_job,
            dep_table=dep_table,
        )

        # HTP ends with ResolveDebugHandle; LPAI runs it immediately before
        # LpaiPartitionFallbackSupport, which must remain the final pass.
        assert (
            tuple(type(p) for p in passes[-len(terminal_passes) :]) == terminal_passes
        )
