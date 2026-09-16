# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch

from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue
from torch.utils import _pytree as pytree


class CortexMQuantizedPass(ExportPass):
    """Base pass for retracing folded Cortex-M quantized graphs.

    Cortex-M's quantized linear kernel includes requantization and therefore
    produces the dtype described by its output qparams, normally int8.

    Once Q/DQ nodes have been folded, retracing ``aten.linear`` through the
    regular PyTorch fake kernel would incorrectly recreate an int32 output.
    Preserve the Cortex-M output dtype instead.
    """

    @staticmethod
    def _is_quantized_meta(meta: NodeMetadata) -> bool:
        input_qparams = meta.data.get("input_qparams", {})
        output_qparams = meta.data.get("output_qparams", {})
        return bool(input_qparams) and bool(output_qparams)

    def _call_quantized_op_without_fake_kernel(
        self,
        op,
        args: tuple[ProxyValue, ...],
        kwargs: dict[str, Any],
        meta: NodeMetadata,
    ) -> ProxyValue:
        old_val = meta.data["val"]
        output_qparams = meta.data.get("output_qparams", {})
        dtype = (
            next(iter(output_qparams.values())).dtype
            if output_qparams
            else old_val.dtype
        )

        result_data = torch.empty_like(old_val, dtype=dtype)

        args_proxy, kwargs_proxy = pytree.tree_map_only(
            ProxyValue,
            lambda value: value.proxy,
            (args, kwargs),
        )

        result_proxy = self.tracer.create_proxy(
            "call_function",
            op,
            args_proxy,
            kwargs_proxy,
        )
        result_proxy.node.meta.update(meta.data)
        self.tracer.set_metadata(result_proxy.node, result_data)

        return ProxyValue(result_data, result_proxy)

    def call_operator(self, op, args, kwargs, meta):
        if (
            op
            in {
                exir_ops.edge.aten.linear.default,
                exir_ops.edge.aten.bmm.default,
            }
            and isinstance(meta, NodeMetadata)
            and self._is_quantized_meta(meta)
        ):
            return self._call_quantized_op_without_fake_kernel(
                op,
                args,
                kwargs,
                meta,
            )

        return super().call_operator(op, args, kwargs, meta)
