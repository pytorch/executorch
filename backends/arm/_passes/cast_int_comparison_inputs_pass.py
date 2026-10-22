# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

import torch

from executorch.backends.arm._passes.arm_pass import ArmOpTargetedPass
from executorch.backends.arm.tosa.cast_support import supported_tosa_casts
from executorch.backends.arm.tosa.specification import (
    get_context_spec,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


class CastIntComparisonInputsPass(ArmOpTargetedPass):
    """Match supported comparison inputs to a dtype supported by the profile.

    Use int32 for integer-only inputs when INT is supported. Otherwise, use
    PyTorch's floating promotion for supported integer/FP mixes, fp16 for
    int8-only inputs, and fp32 for other supported inputs. Reject unsupported
    inputs and casts.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    target_ops = {
        exir_ops.edge.aten.eq.Tensor,
        exir_ops.edge.aten.ne.Tensor,
        exir_ops.edge.aten.ge.Tensor,
        exir_ops.edge.aten.gt.Tensor,
        exir_ops.edge.aten.le.Tensor,
        exir_ops.edge.aten.lt.Tensor,
    }

    def _decide_cast_dtype(
        self, arg_data: list[torch.Tensor], tosa_spec: TosaSpecification
    ) -> torch.dtype:
        """Choose a TOSA comparison dtype from PyTorch's promoted input dtype.

        Use ``torch.result_type`` on the tensors so zero-dimensional inputs
        follow PyTorch promotion. Integer results use int32 when INT is
        available; with FP only, int8 uses fp16 and int16 uses fp32. Floating
        results keep their promoted dtype. The selected dtype must support
        comparisons in the profile, and TOSA must support every required cast.

        Args:
            arg_data (list[torch.Tensor]): Comparison input tensors.
            tosa_spec (TosaSpecification): Active TOSA profile and extensions.

        Returns:
            torch.dtype: Common comparison input dtype.

        Raises:
            ValueError: If the comparison dtype or a required cast is unsupported.

        """
        arg_dtypes = [arg.dtype for arg in arg_data]
        result_dtype = torch.result_type(arg_data[0], arg_data[1])
        int_dtypes = {torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32}

        if tosa_spec.support_integer() and result_dtype in int_dtypes:
            cast_dtype = torch.int32
        elif tosa_spec.support_float() and result_dtype == torch.int8:
            cast_dtype = torch.float16
        elif tosa_spec.support_float() and result_dtype == torch.int16:
            cast_dtype = torch.float32
        else:
            cast_dtype = result_dtype

        comparison_dtypes = set()
        if tosa_spec.support_integer():
            comparison_dtypes.add(torch.int32)
        if tosa_spec.support_float():
            comparison_dtypes.update({torch.float16, torch.float32})
            if tosa_spec.support_extension("bf16"):
                comparison_dtypes.add(torch.bfloat16)

        supported_casts = supported_tosa_casts(tosa_spec)
        if (
            cast_dtype not in comparison_dtypes
            or (
                result_dtype != cast_dtype
                and (result_dtype, cast_dtype) not in supported_casts
            )
            or any(
                dtype != cast_dtype and (dtype, cast_dtype) not in supported_casts
                for dtype in arg_dtypes
            )
        ):
            raise ValueError(
                f"cannot cast comparison input dtypes {arg_dtypes} to a supported dtype."
            )
        return cast_dtype

    def call_operator(self, op, args, kwargs, meta):
        if op not in self.target_ops:
            return super().call_operator(op, args, kwargs, meta)

        arg_dtypes = [arg.data.dtype for arg in args]
        tosa_spec = get_context_spec()

        # Handle cases with bool inputs
        if torch.bool in arg_dtypes:
            if all(dtype == torch.bool for dtype in arg_dtypes) and op in {
                exir_ops.edge.aten.eq.Tensor,
                exir_ops.edge.aten.ne.Tensor,
            }:
                return super().call_operator(op, args, kwargs, meta)

            if not tosa_spec.support_integer() or any(
                dtype
                not in {torch.int8, torch.int16, torch.int32, torch.bool, torch.uint8}
                for dtype in arg_dtypes
            ):
                raise ValueError(
                    f"{op} cannot cast {arg_dtypes} to a {tosa_spec} comparison dtype."
                )

        cast_dtype = self._decide_cast_dtype([arg.data for arg in args], tosa_spec)

        new_args = []
        for arg in args:
            if arg.data.dtype == cast_dtype:
                new_args.append(arg)
            else:
                new_args.append(
                    super().call_operator(
                        exir_ops.edge.dim_order_ops._to_dim_order_copy.default,
                        (arg,),
                        {"dtype": cast_dtype},
                        meta,
                    )
                )
        return super().call_operator(op, tuple(new_args), kwargs, meta)
