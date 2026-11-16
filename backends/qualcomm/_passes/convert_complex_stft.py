# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass

from .utils import copy_meta


class ConvertComplexStft(ExportPass):
    """
    Rewrite a complex-valued stft into its real-valued form so the QNN Stft
    op, whose output is real with the real/imaginary parts in the last
    dimension, can consume it. Only complex consumers that have a real
    equivalent are handled:

        view_as_real(stft(x, return_complex=True)) -> stft(x, return_complex=False)
        abs(stft(x, return_complex=True))          -> sqrt(sum(stft_real ** 2, -1))
        pow(abs(stft(...)), 2)                     -> sum(stft_real ** 2, -1)

    An stft with any other complex consumer is left untouched and is
    decomposed (and runs on CPU) as before.
    """

    stft_op = torch.ops.aten.stft.default
    return_complex_index = 7

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _is_square(node: torch.fx.Node) -> bool:
        return node.target == torch.ops.aten.pow.Tensor_Scalar and node.args[1] == 2

    def _create_node(self, graph, target, args, meta, val):
        node = graph.create_node("call_function", target, args)
        node.meta = copy_meta(meta, callback=lambda m: {**m, "val": val})
        return node

    def _convert(self, graph: torch.fx.Graph, stft: torch.fx.Node) -> None:
        args = list(stft.args)
        args += [None] * (self.return_complex_index + 1 - len(args))
        args[self.return_complex_index] = False
        real_val = torch.view_as_real(stft.meta["val"])
        with graph.inserting_after(stft):
            real_stft = self._create_node(
                graph, self.stft_op, tuple(args), stft.meta, real_val
            )

        for user in list(stft.users):
            if user.target == torch.ops.aten.view_as_real.default:
                # stft traces as a [freq, frames] view over frame-major memory,
                # but the lowered op writes contiguous output; a reshape traces
                # contiguous, so a delegate output keeps the matching dim order.
                with graph.inserting_before(user):
                    contiguous = self._create_node(
                        graph,
                        torch.ops.aten.reshape.default,
                        (real_stft, list(real_val.shape)),
                        user.meta,
                        real_val.contiguous(),
                    )
                user.replace_all_uses_with(contiguous)
                continue
            # The remaining users are abs, as checked by the caller.
            square_val = real_val * real_val
            sum_val = square_val.sum(dim=-1)
            with graph.inserting_before(user):
                square = self._create_node(
                    graph,
                    torch.ops.aten.mul.Tensor,
                    (real_stft, real_stft),
                    user.meta,
                    square_val,
                )
                power = self._create_node(
                    graph,
                    torch.ops.aten.sum.dim_IntList,
                    (square, [-1]),
                    user.meta,
                    sum_val,
                )
            for abs_user in list(user.users):
                if self._is_square(abs_user):
                    abs_user.replace_all_uses_with(power)
            if user.users:
                with graph.inserting_before(user):
                    magnitude = self._create_node(
                        graph,
                        torch.ops.aten.sqrt.default,
                        (power,),
                        user.meta,
                        sum_val.sqrt(),
                    )
                user.replace_all_uses_with(magnitude)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        for node in list(graph.nodes):
            if (
                node.target == self.stft_op
                and node.meta["val"].is_complex()
                and all(
                    user.target
                    in (torch.ops.aten.abs.default, torch.ops.aten.view_as_real.default)
                    for user in node.users
                )
            ):
                self._convert(graph, node)

        dead_code_elimination_pass(graph_module)
        graph_module.recompile()
        return PassResult(graph_module, True)
