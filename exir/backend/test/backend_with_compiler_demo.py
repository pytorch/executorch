# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import final, List, NamedTuple

import torch

from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.exported_program import ExportedProgram


# A simple way to represent an op used in BackendWithCompilerDemo
class DemoOp(NamedTuple):
    op: str
    numel: int
    dtype: str

    def __repr__(self):
        return f"op:demo::{self.op}, numel:{self.numel}, dtype:{self.dtype}"


# Backend details are final (cannot be subclassed).
@final
class BackendWithCompilerDemo(BackendDetails):
    """
    An example implementation to lower a module. Currently this example
    only supports the sin operator.
    The example origin module can be:

    class SinModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sin(x)

    sin_module = SinModule()
    model_inputs = torch.ones(1, 1)

    edgeir_m = to_edge(export(sin_module, model_inputs))
    compile_specs = []
    lowered_sin_module = to_backend(
        "BackendWithCompilerDemo", edgeir_m, compile_specs
    )

    # Module composition of lowered modules is possible.
    class HugeModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lowered_linear_sin = lowered_module

        def forward(self, x):
            output_from_submodule = self.lowered_linear_sin(x)
            return output_from_submodule

    The output trace through graph result is
    graph():
        %arg0_1 : [#users=2] = placeholder[target=arg0_1]
        %lowered_module_0 : [#users=1] = get_attr[target=lowered_module_0]
        %executorch_call_delegate : [#users=1] = call_function[target=torch.ops.higher_order.executorch_call_delegate](args = (%lowered_module_0, forward, %arg0_1), kwargs = {})
        return [executorch_call_delegate]

    Args:
        edge_ir_module: The edge ir module after capture.
        compile_specs: List of backend-specific objects needed for the compilation process

    Returns:
        Bytes: A compiled blob - a binary that can run the desired program in the backend.
    Raises:
        RuntimeError: The module cannot be processed by the backend.
    """

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        processed_bytes = ""
        number_of_instruction = 0
        version = "0"
        debug_handle_map = {}
        match_ops = [
            exir_ops.edge.aten.sin.default,
            exir_ops.edge.aten.mm.default,
            exir_ops.edge.aten.add.Tensor,
            torch.ops.aten.sin.default,
            exir_ops.edge.aten.linear.default,
            exir_ops.edge.aten.scaled_dot_product_attention.default,
            exir_ops.edge.aten.upsample_nearest2d.vec,
        ]

        for node in edge_program.graph.nodes:
            if node.op == "call_function":
                # TODO(gasoonjia): remove the support of torch.ops.aten.sin.default after migrate serde to edge dialect.
                if node.target in match_ops:
                    simple_op = DemoOp(
                        node.target.__name__,
                        int(torch.prod(torch.tensor(node.meta["val"].shape), 0).item()),
                        node.meta["val"].dtype,
                    )
                    number_of_instruction += 1
                    processed_bytes += (
                        str(simple_op)
                        + "<debug_handle>"
                        + str(node.meta.get("debug_handle", -1))
                        + "#"
                    )
                else:
                    raise RuntimeError(
                        f"{node.op} {node.target.__name__} is not supported in backend BackendWithCompilerDemo"
                    )
            elif node.op == "placeholder":
                continue
            elif node.op == "output":
                continue
            elif node.op == "get_attr":
                continue
            else:
                raise RuntimeError(
                    f"{node.op} is not supported in backend BackendWithCompilerDemo"
                )
            # Since the graph remains the same, debug handle remains the same.
            original_debug_id = node.meta["debug_handle"]
            new_debug_id = original_debug_id
            debug_handle_map[new_debug_id] = (original_debug_id,)
        return PreprocessResult(
            processed_bytes=bytes(
                str(number_of_instruction)
                + "version:"
                + version
                + "#"
                + processed_bytes,
                encoding="utf8",
            ),
            debug_handle_map=debug_handle_map,
        )
