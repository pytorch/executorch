from typing import Callable, Dict, Tuple
import torch

import executorch.backends.cortex_m.ops.operators  # noqa

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue


class ReplaceQuantNodesPass(ExportPass):
    """
    Replace quantize and dequantize nodes with the corresponding
    quantize_per_tensor and dequantize_per_tensor nodes.
    """

    @staticmethod
    def is_qualified_quantize_per_tensor(args) -> bool:
        return (
            args[3] >= torch.iinfo(torch.int8).min  # qmin
            and args[4] <= torch.iinfo(torch.int8).max  # qmax
            and args[5] == torch.int8  # output dtype
        )

    @staticmethod
    def is_qualified_dequantize_per_tensor(args) -> bool:
        return (
            args[3] >= torch.iinfo(torch.int8).min  # qmin
            and args[4] <= torch.iinfo(torch.int8).max  # qmax
            and args[5] == torch.int8  # input dtype
        )

    def call_operator(
        self,
        op: Callable[..., object],
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
        meta: NodeMetadata,
    ) -> ProxyValue:
        assert isinstance(
            op, EdgeOpOverload
        ), f"Op must be an EdgeOpOverload, got {type(op)} for op {op}. Try running this pass after to_edge()."
        if (
            op == exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
            and self.is_qualified_quantize_per_tensor(args)
        ):
            return super().call_operator(
                exir_ops.edge.cortex_m.quantize_per_tensor.default,
                args,
                kwargs,
                meta,
            )
        elif (
            op == exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
            and self.is_qualified_dequantize_per_tensor(args)
        ):
            return super().call_operator(
                exir_ops.edge.cortex_m.dequantize_per_tensor.default,
                args,
                kwargs,
                meta,
            )
        # For all other operators, pass through unchanged
        else:
            return super().call_operator(op, args, kwargs, meta)
