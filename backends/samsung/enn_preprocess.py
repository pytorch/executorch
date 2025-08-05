# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, final, List

import executorch.backends.samsung.python.PyEnnWrapperAdaptor as PyEnnWrapper
import torch
from executorch.backends.samsung.serialization.compile_options import (
    ENN_COMPILE_OPTION_TITLE,
)
from executorch.backends.samsung.serialization.enn_graph_schema import EnnGraph

from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)

from executorch.exir.passes import PassManager

from torch.export.exported_program import ExportedProgram


@final
class EnnBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        enn_wrapper = PyEnnWrapper.EnnWrapper()
        option_spec = get_compile_spec(
            compile_specs, ENN_COMPILE_OPTION_TITLE, required=True
        )
        enn_wrapper.Init(option_spec.value)

        enn_preprocess_passes = PassManager(passes=[])

        # 2 make enn graph
        enn_preprocess_passes = PassManager(passes=[])
        pass_result = enn_preprocess_passes(edge_program.graph_module)

        enn_graph = EnnGraph()
        enn_graph.init("UnknownName", "")
        # 3 node visitors
        node_visitors = []

        vals_to_ids: Dict[torch.fx.Node, int] = {}
        for node in pass_result.graph_module.graph.nodes:
            if node.op == "call_function":
                logging.warning(f"Visiting: {node}, {node.target.__name__}")
                if node.target.__name__ in node_visitors:
                    pass
                else:
                    raise RuntimeError(
                        f"{node.target.__name__}" " is not supported in ENN Delegate"
                    )
            elif node.op in [
                "get_attr",
                "placeholder",
                "output",
            ]:
                continue
            else:
                raise RuntimeError(f"{node.op}" " is not supported in ENN Delegate")

        # 4 Compile Graph
        enn_wrapper.Destroy()
        enn_graph.finish()
        ser_buf = enn_graph.serialize()
        enn_context_binary = enn_wrapper.Compile(ser_buf)
        assert enn_context_binary is not None and len(enn_context_binary) > 0
        enn_wrapper.Destroy()
        return PreprocessResult(processed_bytes=bytes(enn_context_binary), debug_handle_map={})
