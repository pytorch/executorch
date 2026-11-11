# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Provide shared dtype checks for TOSA operator support."""

import torch
from executorch.backends.arm.constants import FP_DTYPES, INT_DTYPES
from executorch.backends.arm.tosa import TosaSpecification
from executorch.exir.backend.utils import WhyNoPartitionReporter


def is_indexing_dtype_supported(
    node: torch.fx.Node,
    values_dtype: torch.dtype,
    tosa_spec: TosaSpecification,
    reporter: WhyNoPartitionReporter,
) -> bool:
    """Check value dtypes for indexing against TOSA profiles and extensions.

    Supported integer dtypes and bool require the INT profile. Supported
    floating-point dtypes allow the FP profile or the INT profile via
    quantization. BF16 and FP8 also require their corresponding extensions.

    Args:
        node (torch.fx.Node): Operator node used in rejection messages.
        values_dtype (torch.dtype): Dtype of the values being indexed.
        tosa_spec (TosaSpecification): Active TOSA profiles and extensions.
        reporter (WhyNoPartitionReporter): Reporter for rejection reasons.

    Returns:
        bool: True if the dtype is supported; otherwise, False.

    """
    if values_dtype in INT_DTYPES:
        if not tosa_spec.support_integer():
            reporter.report_reject(
                node,
                f"{node.target}: dtype {values_dtype} requires INT profile.",
            )
            return False
    elif values_dtype in FP_DTYPES:
        if values_dtype == torch.bfloat16 and not tosa_spec.support_extension("bf16"):
            reporter.report_reject(
                node,
                f"{node.target}: dtype {values_dtype} requires bf16 extension.",
            )
            return False
        if values_dtype == torch.float8_e4m3fn and not tosa_spec.support_extension(
            "fp8e4m3"
        ):
            reporter.report_reject(
                node,
                f"{node.target}: dtype {values_dtype} requires fp8e4m3 extension.",
            )
            return False
        if values_dtype == torch.float8_e5m2 and not tosa_spec.support_extension(
            "fp8e5m2"
        ):
            reporter.report_reject(
                node,
                f"{node.target}: dtype {values_dtype} requires fp8e5m2 extension.",
            )
            return False
    else:
        reporter.report_reject(
            node,
            f"{node.target}: unsupported values dtype {values_dtype}; "
            f"expected one of {INT_DTYPES + FP_DTYPES}.",
        )
        return False

    return True
