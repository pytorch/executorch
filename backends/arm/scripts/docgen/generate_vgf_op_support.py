# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Generate backend PyTorch operator-support documentation.

The default output is a customer-facing Markdown page listing public PyTorch
APIs, supported profiles, tested dtypes, and quantization modes. Run the script
from the ExecuTorch repository root.

Options:
    -h, --help           Show command-line help.
    --repo-root PATH     Use PATH as the ExecuTorch repository root.
    --output PATH        Write Markdown to PATH instead of the default page.
    --html               Also write a standalone HTML page beside the Markdown.
    --debug              Include exported ATen operators and contributing tests.
    --check              Validate registry operator/profile coverage; write no page.
    --strict-ast         With --check, fail on unresolved VgfPipeline attribution.
    --explain OP         Explain coverage evidence for one exported ATen operator.

Examples:
    python backends/arm/scripts/docgen/generate_vgf_op_support.py
    python backends/arm/scripts/docgen/generate_vgf_op_support.py --html
    python backends/arm/scripts/docgen/generate_vgf_op_support.py --debug --html --output /tmp/VGF_op_support_debug.md
    python backends/arm/scripts/docgen/generate_vgf_op_support.py --check --strict-ast

"""

from __future__ import annotations

import argparse
import ast
import html
import logging
import operator
import re
import sys

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, Iterable, Mapping, Protocol, Sequence

import torch


# ---------------------------------------------------------------------------
# Backend-specific configuration.
#
# Keep backend-specific names, paths, pipeline signatures, profiles and
# exceptional coverage here. A future Ethos-U variant should be expressible by
# changing this configuration instead of editing the implementation below.
# ---------------------------------------------------------------------------
BACKEND_NAME = "VGF"
BACKEND_PIPELINE_CLASS_NAMES = frozenset({"VgfPipeline"})
BACKEND_PIPELINE_LABEL = "VgfPipeline"
BACKEND_TOSA_SPEC = "TOSA-1.0+FP+INT+int4+int16"

GENERATOR_PATH = Path("backends/arm/scripts/docgen/generate_vgf_op_support.py")
GENERATOR_COMMAND = f"python {GENERATOR_PATH}"
DEFAULT_OUTPUT = Path("docs/source/backends/arm-vgf/VGF_op_support.md")
TEST_ROOT = Path("backends/arm/test")

# Pipeline constructor/static-analysis configuration.
PIPELINE_QUANTIZE_KEYWORD: str | None = "quantize"
PIPELINE_QUANTIZE_DEFAULT = True
PIPELINE_DEFAULT_PROFILE = "INT"
PIPELINE_ATEN_OP_POSITION = 2
PIPELINE_ATEN_OP_KEYWORDS = ("aten_op", "aten_ops")
PIPELINE_EXIR_OP_POSITION = 3
PIPELINE_EXIR_OP_KEYWORDS = ("exir_op", "exir_ops")

PAGE_TITLE = f"PyTorch operator support for the {BACKEND_NAME} backend"
PAGE_DESCRIPTION = (
    f"This page lists {BACKEND_NAME}-supported PyTorch APIs and the dtype and "
    f"quantization modes covered by the {BACKEND_NAME} backend test pipeline."
)
MARKDOWN_DEBUG_NOTE = (
    f"Debug mode adds the exact exported ATen operator and the {BACKEND_NAME} "
    "test functions that contributed each row. Do not publish this version as "
    "the customer-facing page."
)
HTML_DEBUG_NOTE = (
    "Debug mode. This page includes the exact exported ATen operator and the "
    f"{BACKEND_NAME} test functions that contributed each row. Do not publish "
    "it as the customer-facing page."
)
CLI_DESCRIPTION = (
    f"Generate {BACKEND_NAME} PyTorch operator-support documentation from "
    f"{BACKEND_PIPELINE_LABEL} tests."
)

SUPPORT_PROFILE_ORDER = ["FP", "INT"]
DTYPE_ORDER = [
    "FP32",
    "FP16",
    "BF16",
    "FP8E4M3",
    "FP8E5M2",
    "INT8",
    "INT16",
    "INT4",
    "INT32",
    "BOOL",
]
QUANTIZATION_MODE_ORDER = ["8x8", "8x4", "16x8"]

EDGE_OP_ALIASES = {
    "__lshift__.Scalar": "bitwise_left_shift.Scalar",
    "__rshift__.Scalar": "bitwise_right_shift.Scalar",
    "_adaptive_avg_pool2d.default": "adaptive_avg_pool2d.default",
}

# Exact normalisation overrides. Overload-less ATen names are not assumed to
# use the default overload unless listed here.
OVERLOADLESS_OP_ALIASES: dict[str, str] = {
    "torch.ops.aten.amax": "torch.ops.aten.amax.default",
    "torch.ops.aten.amin": "torch.ops.aten.amin.default",
}

# Some legacy tests contain malformed namespaces. Keep the scanner resilient,
# but emit a diagnostic so the test can be corrected.
KNOWN_NAMESPACE_ALIASES = {
    "torch.aten.ops.": "torch.ops.aten.",
}

# Generated EdgeIR spellings are not always reversible by splitting on
# underscores. Check exact aliases before the generic conversion.
GENERATED_EDGE_OP_ALIASES: dict[str, str] = {
    "slice_copy": "slice_copy.Tensor",
    "masked_fill_scalar": "masked_fill.Scalar",
}

# File-specific overload aliases are used only where the test intentionally
# stores an overload-less operator constant.
CONTEXTUAL_OVERLOAD_ALIASES: dict[tuple[str, str], str] = {
    (
        "backends/arm/test/ops/test_amax.py",
        "torch.ops.aten.max",
    ): "torch.ops.aten.max.dim",
    (
        "backends/arm/test/ops/test_amin.py",
        "torch.ops.aten.min",
    ): "torch.ops.aten.min.dim",
}

SOURCE_ATEN = "ATen export"
EDGE_IR = "EdgeIR"
LOWERED = "lowered"
RUNTIME_ONLY = "runtime-only"

DIRECT = "direct"
STAGE_EQUIVALENT = "stage-equivalent"
DECOMPOSED = "decomposed"
TRANSFORM_ONLY = "transform-only"
EXPLICIT = "explicit"
INFERRED = "inferred"

# Registry targets and test assertions can describe different graph stages.
# The key is the backend registry target; values are equivalent ATen/Edge names
# that count as coverage for the same public operation.
STAGE_EQUIVALENT_OPS: dict[str, set[str]] = {
    "torch.ops.aten.convolution.default": {"torch.ops.aten.conv2d.default"},
    "torch.ops.aten._native_batch_norm_legit_no_training.default": {
        "torch.ops.aten.batch_norm.default"
    },
    "torch.ops.aten.constant_pad_nd.default": {"torch.ops.aten.pad.default"},
    "torch.ops.aten.native_group_norm.default": {"torch.ops.aten.group_norm.default"},
    "torch.ops.aten.native_layer_norm.default": {"torch.ops.aten.layer_norm.default"},
    "torch.ops.aten.max_pool2d_with_indices.default": {
        "torch.ops.aten.max_pool2d.default"
    },
    "torch.ops.aten.squeeze_copy.dims": {
        "torch.ops.aten.squeeze.default",
        "torch.ops.aten.squeeze.dim",
        "torch.ops.aten.squeeze.dims",
    },
    "torch.ops.aten.copy.default": {"torch.ops.aten.copy_.default"},
    "torch.ops.aten._softmax.default": {
        "torch.ops.aten.softmax.default",
        "torch.ops.aten.softmax.int",
    },
    "torch.ops.aten._log_softmax.default": {
        "torch.ops.aten.log_softmax.default",
        "torch.ops.aten.log_softmax.int",
    },
    "torch.ops.aten.adaptive_avg_pool2d.default": {"torch.ops.aten.avg_pool2d.default"},
    "torch.ops.aten.slice_copy.Tensor": {"torch.ops.aten.slice.Tensor"},
    "torch.ops.aten.expand_copy.default": {"torch.ops.aten.expand.default"},
    "torch.ops.aten.view_copy.default": {"torch.ops.aten.view.default"},
    "torch.ops.aten.unsqueeze_copy.default": {"torch.ops.aten.unsqueeze.default"},
}

# Quantization can change overloads or decompose a public operator into a
# different ATen operator. These rules are deliberately profile-specific.
PROFILE_STAGE_EQUIVALENT_OPS: dict[tuple[str, str], set[str]] = {
    ("torch.ops.aten.bitwise_and.Scalar", "INT"): {"torch.ops.aten.bitwise_and.Tensor"},
    ("torch.ops.aten.bitwise_or.Scalar", "INT"): {"torch.ops.aten.bitwise_or.Tensor"},
    ("torch.ops.aten.bitwise_xor.Scalar", "INT"): {"torch.ops.aten.bitwise_xor.Tensor"},
    ("torch.ops.aten.bitwise_left_shift.Scalar", "INT"): {
        "torch.ops.aten.bitwise_left_shift.Tensor"
    },
    ("torch.ops.aten.bitwise_right_shift.Scalar", "INT"): {
        "torch.ops.aten.bitwise_right_shift.Tensor"
    },
    ("torch.ops.aten.eq.Scalar", "INT"): {"torch.ops.aten.eq.Tensor"},
    ("torch.ops.aten.ge.Scalar", "INT"): {"torch.ops.aten.ge.Tensor"},
    ("torch.ops.aten.gt.Scalar", "INT"): {"torch.ops.aten.gt.Tensor"},
    ("torch.ops.aten.le.Scalar", "INT"): {"torch.ops.aten.le.Tensor"},
    ("torch.ops.aten.lt.Scalar", "INT"): {"torch.ops.aten.lt.Tensor"},
    ("torch.ops.aten.celu.default", "INT"): {"torch.ops.aten.elu.default"},
    ("torch.ops.aten.selu.default", "INT"): {"torch.ops.aten.elu.default"},
}

# These operators are accepted by partitioning but removed or rewritten before
# backend lowering. They require transformation-path coverage, not a direct backend
# lowering test for every support profile.
TRANSFORM_ONLY_OPS = {
    "torch.ops.aten.alias_copy.default",
    "torch.ops.aten.copy.default",
    "torch.ops.aten.detach_copy.default",
}

# These public operations are intentionally decomposed before backend lowering.
DECOMPOSED_OPS = {
    "torch.ops.aten._softmax.default",
    "torch.ops.aten._log_softmax.default",
    "torch.ops.aten.split_with_sizes_copy.default",
    "torch.ops.aten.native_group_norm.default",
    "torch.ops.aten.native_layer_norm.default",
    "torch.ops.aten.embedding.default",
}


# Tests that intentionally suppress ATen/Edge distribution assertions still
# provide runtime coverage. The key is (relative path, test function).
EXPLICIT_BACKEND_COVERAGE: dict[tuple[str, str], dict[str, set[str]]] = {
    (
        "backends/arm/test/ops/test_div_tensor_mode.py",
        "test_div_tensor_mode_vgf_quant",
    ): {
        "INT": {"torch.ops.aten.div.Tensor_mode"},
    },
    (
        "backends/arm/test/ops/test_masked_fill.py",
        "test_masked_fill_scalar_vgf_no_quant",
    ): {
        "FP": {"torch.ops.aten.masked_fill.Scalar"},
    },
    (
        "backends/arm/test/ops/test_masked_fill.py",
        "test_masked_fill_scalar_vgf_quant",
    ): {
        "INT": {"torch.ops.aten.masked_fill.Scalar"},
    },
    ("backends/arm/test/ops/test_silu.py", "test_silu_vgf_quant"): {
        "INT": {"torch.ops.aten.silu.default"},
    },
    ("backends/arm/test/ops/test_remainder.py", "test_remainder_tensor_vgf_quant"): {
        "INT": {"torch.ops.aten.remainder.Tensor"},
    },
    ("backends/arm/test/ops/test_remainder.py", "test_remainder_scalar_vgf_quant"): {
        "INT": {"torch.ops.aten.remainder.Scalar"},
    },
    ("backends/arm/test/ops/test_var.py", "test_var_dim_vgf_no_quant"): {
        "FP": {"torch.ops.aten.var.dim"},
    },
    ("backends/arm/test/ops/test_var.py", "test_var_dim_vgf_no_quant_no_dim"): {
        "FP": {"torch.ops.aten.var.correction"},
    },
    ("backends/arm/test/ops/test_var.py", "test_var_dim_vgf_no_quant_correction"): {
        "FP": {"torch.ops.aten.var.correction"},
    },
    (
        "backends/arm/test/ops/test_embedding.py",
        "test_embedding_vgf_quant",
    ): {
        "INT": {"torch.ops.aten.embedding.default"},
    },
}

# ---------------------------------------------------------------------------
# Backend-agnostic implementation and shared normalization data.
# ---------------------------------------------------------------------------

# Friendly aliases for exported ATen operators that are commonly produced from
# higher-level PyTorch APIs. These are what should appear in the published page.
# Unknown operators fall back to a conservative ``torch.<aten-name>`` spelling.
PYTORCH_API_ALIASES: dict[str, tuple[str, ...]] = {
    # Internal / higher-order operators with explicit public API spellings.
    "torch.ops.aten._assert_scalar.default": ("torch._assert_scalar",),
    "torch.ops.aten.t_copy.default": (
        "torch.t",
        "torch.Tensor.t",
    ),
    "torch.ops.aten.transpose_copy.int": (
        "torch.transpose",
        "torch.Tensor.transpose",
    ),
    "torch.ops.higher_order.cond": ("torch.cond",),
    "torch.ops.higher_order.while_loop": ("torch.while_loop",),
    # Arithmetic and comparisons.
    "torch.ops.aten.add.Tensor": ("torch.add", "+"),
    "torch.ops.aten.add.Scalar": ("torch.add", "+"),
    "torch.ops.aten.sub.Tensor": ("torch.sub", "-"),
    "torch.ops.aten.sub.Scalar": ("torch.sub", "-"),
    "torch.ops.aten.mul.Tensor": ("torch.mul", "*"),
    "torch.ops.aten.mul.Scalar": ("torch.mul", "*"),
    "torch.ops.aten.div.Tensor": ("torch.div", "/"),
    "torch.ops.aten.div.Scalar": ("torch.div", "/"),
    "torch.ops.aten.floor_divide.default": ("torch.floor_divide", "//"),
    "torch.ops.aten.pow.Tensor_Scalar": ("torch.pow", "**"),
    "torch.ops.aten.pow.Tensor_Tensor": ("torch.pow", "**"),
    "torch.ops.aten.neg.default": ("torch.neg", "unary -"),
    "torch.ops.aten.eq.Tensor": ("torch.eq", "=="),
    "torch.ops.aten.eq.Scalar": ("torch.eq", "=="),
    "torch.ops.aten.ne.Tensor": ("torch.ne", "!="),
    "torch.ops.aten.ne.Scalar": ("torch.ne", "!="),
    "torch.ops.aten.gt.Tensor": ("torch.gt", ">"),
    "torch.ops.aten.gt.Scalar": ("torch.gt", ">"),
    "torch.ops.aten.ge.Tensor": ("torch.ge", ">="),
    "torch.ops.aten.ge.Scalar": ("torch.ge", ">="),
    "torch.ops.aten.lt.Tensor": ("torch.lt", "<"),
    "torch.ops.aten.lt.Scalar": ("torch.lt", "<"),
    "torch.ops.aten.le.Tensor": ("torch.le", "<="),
    "torch.ops.aten.le.Scalar": ("torch.le", "<="),
    "torch.ops.aten.maximum.default": ("torch.maximum",),
    "torch.ops.aten.minimum.default": ("torch.minimum",),
    "torch.ops.aten.clamp.default": ("torch.clamp",),
    "torch.ops.aten.clamp.Tensor": ("torch.clamp",),
    "torch.ops.aten.where.self": ("torch.where",),
    # Bitwise and logical operators.
    "torch.ops.aten.bitwise_and.Tensor": ("torch.bitwise_and", "&"),
    "torch.ops.aten.bitwise_and.Scalar": ("torch.bitwise_and", "&"),
    "torch.ops.aten.bitwise_or.Tensor": ("torch.bitwise_or", "|"),
    "torch.ops.aten.bitwise_or.Scalar": ("torch.bitwise_or", "|"),
    "torch.ops.aten.bitwise_xor.Tensor": ("torch.bitwise_xor", "^"),
    "torch.ops.aten.bitwise_xor.Scalar": ("torch.bitwise_xor", "^"),
    "torch.ops.aten.bitwise_not.default": ("torch.bitwise_not", "~"),
    "torch.ops.aten.bitwise_left_shift.Tensor": ("torch.bitwise_left_shift", "<<"),
    "torch.ops.aten.bitwise_left_shift.Scalar": ("torch.bitwise_left_shift", "<<"),
    "torch.ops.aten.bitwise_right_shift.Tensor": ("torch.bitwise_right_shift", ">>"),
    "torch.ops.aten.bitwise_right_shift.Scalar": ("torch.bitwise_right_shift", ">>"),
    "torch.ops.aten.logical_and.default": ("torch.logical_and",),
    "torch.ops.aten.logical_or.default": ("torch.logical_or",),
    "torch.ops.aten.logical_xor.default": ("torch.logical_xor",),
    "torch.ops.aten.logical_not.default": ("torch.logical_not",),
    # Linear algebra and neural-network layers.
    "torch.ops.aten.linear.default": ("torch.nn.Linear", "torch.nn.functional.linear"),
    "torch.ops.aten.mm.default": ("torch.mm",),
    "torch.ops.aten.bmm.default": ("torch.bmm",),
    "torch.ops.aten.matmul.default": ("torch.matmul", "@"),
    "torch.ops.aten.addmm.default": ("torch.addmm",),
    "torch.ops.aten.convolution.default": (
        "torch.nn.Conv2d",
        "torch.nn.functional.conv2d",
    ),
    "torch.ops.aten.max_pool2d_with_indices.default": (
        "torch.nn.MaxPool2d",
        "torch.nn.functional.max_pool2d",
    ),
    "torch.ops.aten.avg_pool2d.default": (
        "torch.nn.AvgPool2d",
        "torch.nn.functional.avg_pool2d",
    ),
    "torch.ops.aten.adaptive_avg_pool2d.default": (
        "torch.nn.AdaptiveAvgPool2d",
        "torch.nn.functional.adaptive_avg_pool2d",
    ),
    "torch.ops.aten._native_batch_norm_legit_no_training.default": (
        "torch.nn.BatchNorm2d",
        "torch.nn.functional.batch_norm",
    ),
    "torch.ops.aten.native_layer_norm.default": (
        "torch.nn.LayerNorm",
        "torch.nn.functional.layer_norm",
    ),
    "torch.ops.aten.native_group_norm.default": (
        "torch.nn.GroupNorm",
        "torch.nn.functional.group_norm",
    ),
    "torch.ops.aten.embedding.default": (
        "torch.nn.Embedding",
        "torch.nn.functional.embedding",
    ),
    # Activations.
    "torch.ops.aten.relu.default": ("torch.relu", "torch.nn.ReLU"),
    "torch.ops.aten.relu_.default": ("torch.Tensor.relu_",),
    "torch.ops.aten.leaky_relu.default": (
        "torch.nn.LeakyReLU",
        "torch.nn.functional.leaky_relu",
    ),
    "torch.ops.aten.hardtanh.default": (
        "torch.nn.Hardtanh",
        "torch.nn.functional.hardtanh",
    ),
    "torch.ops.aten.gelu.default": ("torch.nn.GELU", "torch.nn.functional.gelu"),
    "torch.ops.aten.hardsigmoid.default": (
        "torch.nn.Hardsigmoid",
        "torch.nn.functional.hardsigmoid",
    ),
    "torch.ops.aten.hardswish.default": (
        "torch.nn.Hardswish",
        "torch.nn.functional.hardswish",
    ),
    "torch.ops.aten.sigmoid.default": ("torch.sigmoid", "torch.nn.Sigmoid"),
    "torch.ops.aten.tanh.default": ("torch.tanh", "torch.nn.Tanh"),
    "torch.ops.aten.silu.default": ("torch.nn.SiLU", "torch.nn.functional.silu"),
    "torch.ops.aten.elu.default": ("torch.nn.ELU", "torch.nn.functional.elu"),
    "torch.ops.aten.selu.default": ("torch.nn.SELU", "torch.nn.functional.selu"),
    "torch.ops.aten.celu.default": ("torch.nn.CELU", "torch.nn.functional.celu"),
    "torch.ops.aten._softmax.default": (
        "torch.nn.Softmax",
        "torch.nn.functional.softmax",
    ),
    "torch.ops.aten._log_softmax.default": (
        "torch.nn.LogSoftmax",
        "torch.nn.functional.log_softmax",
    ),
    "torch.ops.aten.softmax.int": ("torch.nn.Softmax", "torch.nn.functional.softmax"),
    "torch.ops.aten.log_softmax.int": (
        "torch.nn.LogSoftmax",
        "torch.nn.functional.log_softmax",
    ),
    # Reductions.
    "torch.ops.aten.sum.default": ("torch.sum",),
    "torch.ops.aten.sum.dim_IntList": ("torch.sum",),
    "torch.ops.aten.mean.default": ("torch.mean",),
    "torch.ops.aten.mean.dim": ("torch.mean",),
    "torch.ops.aten.amax.default": ("torch.amax",),
    "torch.ops.aten.amin.default": ("torch.amin",),
    "torch.ops.aten.max.dim": ("torch.max",),
    "torch.ops.aten.min.dim": ("torch.min",),
    "torch.ops.aten.argmax.default": ("torch.argmax",),
    "torch.ops.aten.argmax.dim": ("torch.argmax",),
    "torch.ops.aten.var.correction": ("torch.var",),
    "torch.ops.aten.var.dim": ("torch.var",),
    "torch.ops.aten.any.default": ("torch.any",),
    "torch.ops.aten.any.dim": ("torch.any",),
    "torch.ops.aten.any.dims": ("torch.any",),
    # Elementwise math.
    "torch.ops.aten.abs.default": ("torch.abs",),
    "torch.ops.aten.exp.default": ("torch.exp",),
    "torch.ops.aten.expm1.default": ("torch.expm1",),
    "torch.ops.aten.log.default": ("torch.log",),
    "torch.ops.aten.log1p.default": ("torch.log1p",),
    "torch.ops.aten.log10.default": ("torch.log10",),
    "torch.ops.aten.logit.default": ("torch.logit",),
    "torch.ops.aten.sqrt.default": ("torch.sqrt",),
    "torch.ops.aten.rsqrt.default": ("torch.rsqrt",),
    "torch.ops.aten.reciprocal.default": ("torch.reciprocal",),
    "torch.ops.aten.floor.default": ("torch.floor",),
    "torch.ops.aten.ceil.default": ("torch.ceil",),
    "torch.ops.aten.round.default": ("torch.round",),
    "torch.ops.aten.sin.default": ("torch.sin",),
    "torch.ops.aten.sinh.default": ("torch.sinh",),
    "torch.ops.aten.cos.default": ("torch.cos",),
    "torch.ops.aten.cosh.default": ("torch.cosh",),
    "torch.ops.aten.tan.default": ("torch.tan",),
    "torch.ops.aten.atan.default": ("torch.atan",),
    "torch.ops.aten.atanh.default": ("torch.atanh",),
    "torch.ops.aten.asin.default": ("torch.asin",),
    "torch.ops.aten.asinh.default": ("torch.asinh",),
    "torch.ops.aten.acos.default": ("torch.acos",),
    "torch.ops.aten.acosh.default": ("torch.acosh",),
    "torch.ops.aten.erf.default": ("torch.erf",),
    "torch.ops.aten.erfinv.default": ("torch.erfinv",),
    "torch.ops.aten.remainder.Scalar": ("torch.remainder",),
    "torch.ops.aten.remainder.Tensor": ("torch.remainder",),
    # Tensor creation / movement / shape operators.
    "torch.ops.aten.cat.default": ("torch.cat",),
    "torch.ops.aten.stack.default": ("torch.stack",),
    "torch.ops.aten.view.default": ("torch.Tensor.view",),
    "torch.ops.aten.view_copy.default": ("torch.Tensor.view",),
    "torch.ops.aten.reshape.default": ("torch.reshape", "torch.Tensor.reshape"),
    "torch.ops.aten.permute.default": ("torch.permute", "torch.Tensor.permute"),
    "torch.ops.aten.permute_copy.default": ("torch.permute", "torch.Tensor.permute"),
    "torch.ops.aten.transpose.int": ("torch.transpose", "torch.Tensor.transpose"),
    "torch.ops.aten.squeeze.default": ("torch.squeeze", "torch.Tensor.squeeze"),
    "torch.ops.aten.squeeze.dim": ("torch.squeeze", "torch.Tensor.squeeze"),
    "torch.ops.aten.squeeze_copy.dims": ("torch.squeeze", "torch.Tensor.squeeze"),
    "torch.ops.aten.unsqueeze.default": ("torch.unsqueeze", "torch.Tensor.unsqueeze"),
    "torch.ops.aten.unsqueeze_copy.default": (
        "torch.unsqueeze",
        "torch.Tensor.unsqueeze",
    ),
    "torch.ops.aten.expand.default": ("torch.Tensor.expand",),
    "torch.ops.aten.expand_copy.default": ("torch.Tensor.expand",),
    "torch.ops.aten.repeat.default": ("torch.Tensor.repeat",),
    "torch.ops.aten.clone.default": ("torch.clone", "torch.Tensor.clone"),
    "torch.ops.aten.alias_copy.default": ("torch.alias_copy",),
    "torch.ops.aten.detach_copy.default": ("torch.detach", "torch.Tensor.detach"),
    "torch.ops.aten.copy.default": ("torch.Tensor.copy_",),
    "torch.ops.aten.contiguous.default": ("torch.Tensor.contiguous",),
    "torch.ops.aten.to.dtype": ("torch.Tensor.to",),
    "torch.ops.aten._to_copy.default": ("torch.Tensor.to",),
    "torch.ops.aten.full.default": ("torch.full",),
    "torch.ops.aten.full_like.default": ("torch.full_like",),
    "torch.ops.aten.zeros.default": ("torch.zeros",),
    "torch.ops.aten.zeros_like.default": ("torch.zeros_like",),
    "torch.ops.aten.ones.default": ("torch.ones",),
    "torch.ops.aten.ones_like.default": ("torch.ones_like",),
    "torch.ops.aten.eye.default": ("torch.eye",),
    "torch.ops.aten.arange.start_step": ("torch.arange",),
    "torch.ops.aten.linspace.default": ("torch.linspace",),
    "torch.ops.aten.scalar_tensor.default": ("torch.scalar_tensor",),
    "torch.ops.aten.split_copy.Tensor": ("torch.split", "torch.Tensor.split"),
    "torch.ops.aten.split_with_sizes_copy.default": (
        "torch.split",
        "torch.Tensor.split",
    ),
    "torch.ops.aten.unfold_copy.default": ("torch.Tensor.unfold",),
    "torch.ops.aten.slice_scatter.default": ("torch.slice_scatter",),
    # Indexing and data movement.
    "torch.ops.aten.select.int": ("torch.select", "torch.Tensor.select"),
    "torch.ops.aten.select_copy.int": ("torch.select", "torch.Tensor.select"),
    "torch.ops.aten.slice.Tensor": ("torch.Tensor.__getitem__", "tensor slicing"),
    "torch.ops.aten.slice_copy.Tensor": ("torch.Tensor.__getitem__", "tensor slicing"),
    "torch.ops.aten.index.Tensor": ("torch.Tensor.__getitem__", "tensor indexing"),
    "torch.ops.aten.gather.default": ("torch.gather",),
    "torch.ops.aten.index_select.default": ("torch.index_select",),
    "torch.ops.aten.index_put.default": (
        "torch.Tensor.__setitem__",
        "tensor indexing assignment",
    ),
    "torch.ops.aten.masked_fill.Scalar": (
        "torch.masked_fill",
        "torch.Tensor.masked_fill",
    ),
    "torch.ops.aten.constant_pad_nd.default": ("torch.nn.functional.pad",),
    "torch.ops.aten.pad.default": ("torch.nn.functional.pad",),
    "torch.ops.aten.flip.default": ("torch.flip",),
}

GENERATED_EDGE_OVERLOAD_SUFFIXES = [
    "_dim_IntList",
    "_Tensor_Scalar",
    "_Tensor_Tensor",
    "_Scalar_Tensor",
    "_Tensor",
    "_Scalar",
    "_dtype",
    "_int",
    "_self",
    "_dim",
    "_dims",
    "_default",
]

MAX_STATIC_EXPANSION = 256

logger = logging.getLogger(__name__)


class TosaSpecificationLike(Protocol):
    """Structural type for the TOSA profile queries used by this script."""

    def support_float(self) -> bool: ...

    def support_integer(self) -> bool: ...


@dataclass(frozen=True)
class CoverageEvidence:
    exported_op: str
    profile: str
    stage: str
    classification: str
    test: str
    asserted_op: str | None = None


@dataclass
class UnresolvedPipelineEvidence:
    path: Path
    function: str
    profile: str
    aten_expression: str
    exir_expression: str
    reason: str


@dataclass(frozen=True)
class StaticBinding:
    expression: ast.AST
    scope_locals: dict[str, list[ast.AST]]


@dataclass
class PipelineCoverage:
    """Coverage inferred from one or more configured backend pipeline tests."""

    exported_op: str
    pytorch_apis: tuple[str, ...]
    support_profiles: set[str] = field(default_factory=set)
    dtypes: set[str] = field(default_factory=set)
    quantization_modes: set[str] = field(default_factory=set)
    tests: set[str] = field(default_factory=set)
    stages: set[str] = field(default_factory=set)
    classifications: set[str] = field(default_factory=set)
    evidence_records: list[CoverageEvidence] = field(default_factory=list)


@dataclass
class PublicCoverage:
    """Customer-facing coverage aggregated by PyTorch API spelling."""

    pytorch_apis: tuple[str, ...]
    support_profiles: set[str] = field(default_factory=set)
    dtypes: set[str] = field(default_factory=set)
    quantization_modes: set[str] = field(default_factory=set)


@dataclass
class SupportedOperatorEvidence:
    """Backend evidence that an exported ATen operator should have tests."""

    exported_op: str
    pytorch_apis: tuple[str, ...]
    support_profiles: set[str] = field(default_factory=set)
    evidence: set[str] = field(default_factory=set)


@dataclass
class ClassContext:
    """Static information extracted from one Python class."""

    assignments: dict[str, ast.AST] = field(default_factory=dict)
    methods: dict[str, ast.FunctionDef] = field(default_factory=dict)
    fields: list[str] = field(default_factory=list)


@dataclass
class ModuleContext:
    """Static information extracted from a Python test module."""

    path: Path
    source: str
    tree: ast.Module
    assignments: dict[str, ast.AST]
    functions: dict[str, ast.FunctionDef]
    classes: dict[str, ClassContext]
    parents: dict[ast.AST, ast.AST]


@dataclass
class ResolutionScope:
    """Local static bindings available at a particular call site."""

    locals: dict[str, list[ast.AST]] = field(default_factory=dict)
    param_candidates: dict[str, list[ast.AST]] = field(default_factory=dict)
    class_name: str | None = None


def _scope_with_class(scope: ResolutionScope, class_name: str) -> ResolutionScope:
    return ResolutionScope(
        locals=scope.locals,
        param_candidates=scope.param_candidates,
        class_name=class_name,
    )


def _repo_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        return repo_root.resolve()

    candidates = [Path.cwd().resolve(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "backends/arm").is_dir() and (
            candidate / "docs/source"
        ).is_dir():
            return candidate

    raise RuntimeError("Could not locate the ExecuTorch repository root.")


def _ensure_repo_importable(repo_root: Path) -> None:
    root = str(repo_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def _format_backtick_items(
    items: Iterable[str], order: Sequence[str] | None = None
) -> str:
    values = _sort_items(items, order)
    return ", ".join(f"`{value}`" for value in values) if values else "-"


def _format_items(items: Iterable[str], order: Sequence[str] | None = None) -> str:
    values = _sort_items(items, order)
    return ", ".join(values) if values else "-"


def _format_api_items(items: Sequence[str]) -> str:
    return " / ".join(f"`{item}`" for item in items) if items else "-"


def _format_test_items(items: Iterable[str]) -> str:
    values = sorted(set(items))
    return "<br />".join(f"`{value}`" for value in values) if values else "-"


def _html_code(value: str) -> str:
    return f"<code>{html.escape(value)}</code>"


def _format_html_code_items(
    items: Iterable[str], order: Sequence[str] | None = None
) -> str:
    values = _sort_items(items, order)
    return ", ".join(_html_code(value) for value in values) if values else "-"


def _format_html_items(items: Iterable[str], order: Sequence[str] | None = None) -> str:
    values = _sort_items(items, order)
    return ", ".join(html.escape(value) for value in values) if values else "-"


def _format_html_api_items(items: Sequence[str]) -> str:
    return " / ".join(_html_code(item) for item in items) if items else "-"


def _format_html_test_items(items: Iterable[str]) -> str:
    values = sorted(set(items))
    return "<br />".join(_html_code(value) for value in values) if values else "-"


def _sort_items(items: Iterable[str], order: Sequence[str] | None = None) -> list[str]:
    values = {item for item in items if item}
    if order is None:
        return sorted(values)
    position = {item: idx for idx, item in enumerate(order)}
    return sorted(values, key=lambda item: (position.get(item, 10_000), item))


def _expr_text(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _attribute_name(node: ast.AST) -> str:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _call_name(node.value)
    return ""


def _is_backend_pipeline_call(node: ast.Call) -> bool:
    return _call_name(node.func) in BACKEND_PIPELINE_CLASS_NAMES


def _literal_bool(node: ast.AST | None, default: bool) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return default


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _positional_or_keyword(call: ast.Call, index: int, name: str) -> ast.AST | None:
    if len(call.args) > index:
        return call.args[index]
    return _keyword(call, name)


def _pipeline_profile(call: ast.Call) -> str:
    if PIPELINE_QUANTIZE_KEYWORD is None:
        return PIPELINE_DEFAULT_PROFILE
    quantize = _literal_bool(
        _keyword(call, PIPELINE_QUANTIZE_KEYWORD),
        default=PIPELINE_QUANTIZE_DEFAULT,
    )
    return "INT" if quantize else "FP"


def _pipeline_operator_expr(
    call: ast.Call, position: int, keyword_names: tuple[str, ...]
) -> ast.AST | None:
    if not keyword_names:
        return call.args[position] if len(call.args) > position else None

    value = _positional_or_keyword(call, position, keyword_names[0])
    if value is not None:
        return value
    for keyword_name in keyword_names[1:]:
        value = _keyword(call, keyword_name)
        if value is not None:
            return value
    return None


def _edge_generated_name_to_aten(text: str) -> str:
    exact = GENERATED_EDGE_OP_ALIASES.get(text)
    if exact is not None:
        return exact
    for suffix in GENERATED_EDGE_OVERLOAD_SUFFIXES:
        if text.endswith(suffix):
            return f"{text[: -len(suffix)]}.{suffix[1:]}"
    return text.replace("_", ".", 1)


def _normalize_pytorch_op_name(  # noqa: C901
    op: str,
    *,
    path: Path | None = None,
    diagnostics: list[str] | None = None,
) -> str | None:
    """Normalize test and registry spellings to canonical exported ATen
    names.
    """

    text = op.strip().strip("'\"")
    if not text or text in {"[]", "None"}:
        return None
    if "quantized_decomposed" in text:
        return None
    if text in {"operator.getitem", "<built-in function getitem>"}:
        return None
    if text.startswith("tosa.") or ".tosa." in text:
        return None

    for bad_prefix, good_prefix in KNOWN_NAMESPACE_ALIASES.items():
        if text.startswith(bad_prefix):
            if diagnostics is not None:
                diagnostics.append(f"normalised malformed namespace: {text}")
            text = good_prefix + text.removeprefix(bad_prefix)
            break

    generated_prefix = "executorch_exir_dialects_edge__ops_aten_"
    recognized = (
        text.startswith(generated_prefix)
        or text.startswith("torch.ops.")
        or text.startswith("exir_ops.edge.")
        or text.startswith("executorch.exir.dialects._ops.ops.edge.")
        or text.startswith("edge.")
        or text.startswith("aten::")
        or text.startswith("aten.")
        or ".aten." in text
    )
    if not recognized:
        return None

    original = text

    # Higher-order operators are not ATen operators. Preserve their namespace
    # instead of rewriting torch.ops.higher_order.* as torch.ops.aten.*.
    if text.startswith("torch.ops.higher_order."):
        return text

    if text.startswith(generated_prefix):
        text = _edge_generated_name_to_aten(text.removeprefix(generated_prefix))

    for prefix in (
        "torch.ops.",
        "exir_ops.edge.",
        "executorch.exir.dialects._ops.ops.edge.",
        "edge.",
    ):
        if text.startswith(prefix):
            text = text.removeprefix(prefix)

    if text.startswith("aten::"):
        text = text.removeprefix("aten::")
        if "." not in text:
            text = f"{text}.default"
    elif text.startswith("aten."):
        text = text.removeprefix("aten.")
    elif ".aten." in text:
        text = text.split(".aten.", maxsplit=1)[1]

    text = EDGE_OP_ALIASES.get(text, text)
    canonical_without_overload = f"torch.ops.aten.{text}"
    if "." not in text:
        contextual = None
        if path is not None:
            contextual = CONTEXTUAL_OVERLOAD_ALIASES.get(
                (str(path), canonical_without_overload)
            )
        if contextual is not None:
            return contextual
        explicit = OVERLOADLESS_OP_ALIASES.get(canonical_without_overload)
        if explicit is not None:
            return explicit
        if diagnostics is not None:
            diagnostics.append(f"ambiguous overload-less operator: {original}")
        return None

    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z0-9_]+$", text):
        return None
    return f"torch.ops.aten.{text}"


def _canonical_pytorch_op_from_target(target: object) -> str | None:
    if target is operator.getitem:
        return None

    schema = getattr(target, "_schema", None)
    if schema is not None:
        schema_name = str(getattr(schema, "name", ""))
        overload = str(getattr(schema, "overload_name", "")) or "default"
        namespace, _, op_name = schema_name.partition("::")
        if namespace != "aten" or not op_name:
            return None
        return _normalize_pytorch_op_name(f"aten.{op_name}.{overload}")

    return _normalize_pytorch_op_name(str(target))


def _split_exported_op(exported_op: str) -> tuple[str, str] | None:
    prefix = "torch.ops.aten."
    if not exported_op.startswith(prefix):
        return None
    body = exported_op.removeprefix(prefix)
    if "." not in body:
        return None
    name, overload = body.rsplit(".", maxsplit=1)
    return name, overload


def _torch_api_exists(api: str) -> bool:
    """Return whether a dotted torch API exists and is callable."""

    if not api.startswith("torch."):
        return False

    obj: object = torch
    for part in api.split(".")[1:]:
        if not hasattr(obj, part):
            return False
        obj = getattr(obj, part)

    return callable(obj)


def _fallback_pytorch_api_aliases(exported_op: str) -> tuple[str, ...]:
    split = _split_exported_op(exported_op)
    if split is None:
        return ()

    name, _overload = split
    candidate = f"torch.{name}"
    return (candidate,) if _torch_api_exists(candidate) else ()


def _pytorch_api_aliases(exported_op: str) -> tuple[str, ...]:
    return PYTORCH_API_ALIASES.get(
        exported_op, _fallback_pytorch_api_aliases(exported_op)
    )


def _api_sort_key(apis: Sequence[str]) -> tuple[str, ...]:
    return tuple(api.lower() for api in apis)


def _api_key(apis: Sequence[str]) -> str:
    return "\0".join(apis)


def _call_class_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _call_keyword_bool(call: ast.Call, name: str, default: bool = False) -> bool:
    value = _keyword(call, name)
    return _literal_bool(value, default)


def _class_attr_value(
    class_name: str,
    attr: str,
    context: ModuleContext,
    seen: set[tuple[str, str]] | None = None,
) -> ast.AST | None:
    class_ctx = context.classes.get(class_name)
    if class_ctx is None:
        return None

    key = (class_name, attr)
    if seen is None:
        seen = set()
    if key in seen:
        return None
    seen.add(key)

    value = class_ctx.assignments.get(attr)
    if isinstance(value, ast.Name):
        if value.id == attr:
            # In a class body, ``aten_op = aten_op`` resolves the right-hand
            # name from the surrounding module because the class attribute is
            # not bound until the assignment completes.
            return context.assignments.get(value.id, value)
        if value.id in class_ctx.assignments:
            # Resolve class-local aliases such as
            # ``quantized_aten_op = aten_op`` before falling back to module
            # assignments.
            return _class_attr_value(class_name, value.id, context, seen)
    return value


def _resolve_name_expr(
    name: str,
    context: ModuleContext,
    scope: ResolutionScope,
    seen: set[tuple[str, str]],
) -> list[ast.AST]:
    """Resolve a name to one or more candidate AST expressions."""

    if scope.class_name is not None:
        class_value = _class_attr_value(scope.class_name, name, context)
        if class_value is not None:
            class_key = ("class_attr", f"{scope.class_name}.{name}")
            if class_key in seen:
                return []
            seen.add(class_key)
            return _candidate_exprs(class_value, context, scope, seen)

    key = ("name", name)
    if key in seen:
        return []
    seen.add(key)

    if name in scope.locals:
        values: list[ast.AST] = []
        for candidate in scope.locals[name]:
            values.extend(_candidate_exprs(candidate, context, scope, seen.copy()))
        return values[:MAX_STATIC_EXPANSION]
    if name in scope.param_candidates:
        return scope.param_candidates[name]
    if name in context.assignments:
        return _candidate_exprs(context.assignments[name], context, scope, seen)
    return []


def _dict_value_candidates(node: ast.Dict) -> list[ast.AST]:
    # common.parametrize(name, dict) in the Arm tests parametrizes over values.
    # Keys are test IDs; values are lambdas or pytest.param objects.
    return [value for value in node.values if value is not None]


def _parametrize_names(node: ast.AST) -> list[str]:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        return []
    return [part.strip() for part in node.value.split(",") if part.strip()]


def _candidate_exprs(  # noqa: C901
    node: ast.AST | None,
    context: ModuleContext,
    scope: ResolutionScope,
    seen: set[tuple[str, str]] | None = None,
) -> list[ast.AST]:
    """Expand an AST expression into possible runtime values when feasible."""

    if node is None:
        return []
    if seen is None:
        seen = set()

    if isinstance(node, ast.Name):
        name_values = _resolve_name_expr(node.id, context, scope, seen)
        return name_values or [node]

    if isinstance(node, ast.Lambda):
        return _candidate_exprs(node.body, context, scope, seen)

    if isinstance(node, ast.Call):
        call_name = _call_name(node.func)
        if call_name == "param" and node.args:
            return _candidate_exprs(node.args[0], context, scope, seen)
        if isinstance(node.func, ast.Name) and node.func.id in scope.param_candidates:
            parameter_values: list[ast.AST] = []
            for candidate in scope.param_candidates[node.func.id]:
                parameter_values.extend(
                    _candidate_exprs(candidate, context, scope, seen)
                )
            return parameter_values[:MAX_STATIC_EXPANSION]
        return [node]

    if isinstance(node, ast.Dict):
        dictionary_values: list[ast.AST] = []
        for value in _dict_value_candidates(node):
            dictionary_values.extend(_candidate_exprs(value, context, scope, seen))
        return dictionary_values[:MAX_STATIC_EXPANSION]

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        union_values = _candidate_exprs(
            node.left, context, scope, seen
        ) + _candidate_exprs(node.right, context, scope, seen)
        return union_values[:MAX_STATIC_EXPANSION]

    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        collection_values: list[ast.AST] = []
        for element in node.elts:
            collection_values.extend(_candidate_exprs(element, context, scope, seen))
        return collection_values[:MAX_STATIC_EXPANSION]

    return [node]


def _parametrize_case_exprs(  # noqa: C901
    node: ast.AST | None,
    context: ModuleContext,
    scope: ResolutionScope,
    seen: set[tuple[str, str]] | None = None,
) -> list[ast.AST]:
    """Expand a parametrization source while preserving each case tuple."""

    if node is None:
        return []
    if seen is None:
        seen = set()
    if isinstance(node, ast.Name):
        key = ("param-source", node.id)
        if key in seen:
            return []
        seen.add(key)
        if node.id in context.assignments:
            return _parametrize_case_exprs(
                context.assignments[node.id], context, scope, seen
            )
        return [node]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return (
            _parametrize_case_exprs(node.left, context, scope, seen.copy())
            + _parametrize_case_exprs(node.right, context, scope, seen.copy())
        )[:MAX_STATIC_EXPANSION]
    if isinstance(node, ast.Dict):
        dictionary_cases: list[ast.AST] = []
        for value in _dict_value_candidates(node):
            dictionary_cases.extend(
                _parametrize_case_exprs(value, context, scope, seen.copy())
            )
        return dictionary_cases[:MAX_STATIC_EXPANSION]
    if isinstance(node, ast.Lambda):
        return [node.body]
    if isinstance(node, ast.Call) and _call_name(node.func) == "param" and node.args:
        return [node.args[0]]
    if isinstance(node, (ast.List, ast.Set)):
        collection_cases: list[ast.AST] = []
        for element in node.elts:
            collection_cases.extend(
                _parametrize_case_exprs(element, context, scope, seen.copy())
            )
        return collection_cases[:MAX_STATIC_EXPANSION]
    # A tuple is one case for single-argument parametrization. Multi-argument
    # parametrization splits it later in _parameter_candidates.
    return [node]


def _parameter_candidates(
    function: ast.FunctionDef, context: ModuleContext
) -> dict[str, list[ast.AST]]:
    result: dict[str, list[ast.AST]] = {}
    scope = ResolutionScope()

    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        if "parametrize" not in _expr_text(decorator.func):
            continue
        if len(decorator.args) < 2:
            continue
        names = _parametrize_names(decorator.args[0])
        if not names:
            continue

        candidates = _parametrize_case_exprs(decorator.args[1], context, scope)
        if len(names) == 1:
            result.setdefault(names[0], []).extend(candidates[:MAX_STATIC_EXPANSION])
            continue

        for candidate in candidates:
            if not isinstance(candidate, (ast.Tuple, ast.List)):
                continue
            for name, value in zip(names, candidate.elts):
                result.setdefault(name, []).append(value)

    for name, candidates in result.items():
        result[name] = candidates[:MAX_STATIC_EXPANSION]
    return result


def _bind_assignment_target(
    target: ast.AST,
    values: list[ast.AST],
    bindings: dict[str, list[ast.AST]],
) -> None:
    if isinstance(target, ast.Name):
        bindings[target.id] = values[:MAX_STATIC_EXPANSION]
        return

    if isinstance(target, ast.Starred):
        _bind_assignment_target(target.value, values, bindings)
        return
    if not isinstance(target, (ast.Tuple, ast.List)):
        return

    element_values: list[list[ast.AST]] = [[] for _ in target.elts]
    for value in values:
        if not isinstance(value, (ast.Tuple, ast.List)):
            continue
        for index, element in enumerate(value.elts[: len(target.elts)]):
            element_values[index].append(element)

    for target_element, candidates in zip(target.elts, element_values):
        if candidates:
            _bind_assignment_target(target_element, candidates, bindings)


def _iter_statements_before(
    statements: list[ast.stmt], marker_line: int
) -> Iterable[ast.stmt]:
    """Yield statements lexically preceding a marker, including nested blocks.

    Compound statement nodes do not expose a uniform set of fields. In
    particular, ``with``/``async with`` have ``body`` but no ``orelse``. Keep
    the cases explicit rather than probing attributes that do not exist.

    Nested function and class bodies are intentionally not traversed: their
    assignments are not local bindings executed in the enclosing test function.

    """

    for statement in statements:
        if getattr(statement, "lineno", 10**9) >= marker_line:
            continue

        yield statement
        nested: list[list[ast.stmt]] = []

        if isinstance(statement, ast.If):
            nested.extend((statement.body, statement.orelse))
        elif isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
            nested.extend((statement.body, statement.orelse))
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            nested.append(statement.body)
        elif isinstance(statement, ast.Try):
            nested.extend((statement.body, statement.orelse, statement.finalbody))
            nested.extend(handler.body for handler in statement.handlers)
        elif type(statement).__name__ == "TryStar":
            # ``ast.TryStar`` is available only on newer Python versions. The
            # class-name check does not narrow ``statement`` for MyPy, so use
            # targeted ignores on its try-specific fields.
            nested.extend(
                (
                    cast(
                        list[ast.stmt],
                        statement.body,  # type: ignore[attr-defined]
                    ),
                    cast(
                        list[ast.stmt],
                        statement.orelse,  # type: ignore[attr-defined]
                    ),
                    cast(
                        list[ast.stmt],
                        statement.finalbody,  # type: ignore[attr-defined]
                    ),
                )
            )
            handlers = cast(
                list[ast.ExceptHandler],
                statement.handlers,  # type: ignore[attr-defined]
            )
            nested.extend(handler.body for handler in handlers)
        elif isinstance(statement, ast.Match):
            nested.extend(case.body for case in statement.cases)

        for block in nested:
            yield from _iter_statements_before(block, marker_line)


def _local_assignments_before(
    function: ast.FunctionDef,
    marker: ast.AST,
    context: ModuleContext,
    param_candidates: dict[str, list[ast.AST]],
) -> dict[str, list[ast.AST]]:
    """Collect local bindings in source order using pre-assignment RHS scope."""

    marker_line = getattr(marker, "lineno", 10**9)
    bindings: dict[str, list[ast.AST]] = {}
    for node in _iter_statements_before(function.body, marker_line):
        if isinstance(node, ast.Assign):
            snapshot = {name: list(values) for name, values in bindings.items()}
            scope = ResolutionScope(locals=snapshot, param_candidates=param_candidates)
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in param_candidates
            ):
                values = list(param_candidates[node.value.func.id])
            else:
                values = _candidate_exprs(node.value, context, scope) or [node.value]
            for target in node.targets:
                _bind_assignment_target(target, values, bindings)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            snapshot = {name: list(values) for name, values in bindings.items()}
            scope = ResolutionScope(locals=snapshot, param_candidates=param_candidates)
            values = _candidate_exprs(node.value, context, scope) or [node.value]
            _bind_assignment_target(node.target, values, bindings)
    return bindings


def _scope_for_call(
    function: ast.FunctionDef, call: ast.Call, context: ModuleContext
) -> ResolutionScope:
    params = _parameter_candidates(function, context)
    return ResolutionScope(
        locals=_local_assignments_before(function, call, context, params),
        param_candidates=params,
    )


def _constructor_arg(
    call: ast.Call,
    class_ctx: ClassContext,
    field_name: str,
) -> ast.AST | None:
    if field_name in class_ctx.fields:
        index = class_ctx.fields.index(field_name)
        if len(call.args) > index:
            return call.args[index]

    for keyword in call.keywords:
        if keyword.arg == field_name:
            return keyword.value
    return None


def _model_attr_from_test_case_call(
    test_case_call: ast.Call,
    attr: str,
    context: ModuleContext,
    scope: ResolutionScope,
) -> list[str]:
    """Resolve patterns such as EluTestCase(Elu(), ...).aten_op()."""

    class_name = _call_class_name(test_case_call)
    if class_name is None:
        return []

    class_ctx = context.classes.get(class_name)
    if class_ctx is None:
        return []

    model_expr = _constructor_arg(test_case_call, class_ctx, "model")
    if model_expr is None and test_case_call.args:
        # Dataclass test case helpers nearly always place the module/model as
        # the first field. Use that as a conservative fallback.
        model_expr = test_case_call.args[0]
    if model_expr is None:
        return []

    values: list[str] = []
    for model_candidate in _candidate_exprs(model_expr, context, scope):
        if isinstance(model_candidate, ast.Call):
            model_class = _call_class_name(model_candidate)
        elif isinstance(model_candidate, ast.Name):
            model_class = model_candidate.id
        else:
            model_class = None
        if model_class is None:
            continue

        attr_value = _class_attr_value(model_class, attr, context)
        values.extend(
            _op_texts(attr_value, context, _scope_with_class(scope, model_class))
        )
    return values


def _resolve_getattr_call(
    node: ast.Call,
    context: ModuleContext,
    scope: ResolutionScope,
) -> list[str]:
    """Resolve simple getattr(obj, "aten_op") static patterns."""

    if _call_name(node.func) != "getattr" or len(node.args) < 2:
        return []

    attr_names: list[str] = []
    attr_arg = node.args[1]
    if isinstance(attr_arg, ast.Constant) and isinstance(attr_arg.value, str):
        attr_names = [attr_arg.value]
    elif isinstance(attr_arg, ast.Name):
        for candidate in _candidate_exprs(attr_arg, context, scope):
            if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
                attr_names.append(candidate.value)
    if not attr_names:
        return []

    values: list[str] = []
    for obj_candidate in _candidate_exprs(node.args[0], context, scope):
        if isinstance(obj_candidate, ast.Call):
            class_name = _call_class_name(obj_candidate)
        elif isinstance(obj_candidate, ast.Name):
            class_name = obj_candidate.id
        else:
            class_name = None
        if class_name is None:
            continue
        for attr_name in attr_names:
            values.extend(
                _op_texts(
                    _class_attr_value(class_name, attr_name, context),
                    context,
                    _scope_with_class(scope, class_name),
                )
            )
    return values


def _method_attr_for_call(method_name: str, call: ast.Call) -> str | None:
    """Map helper method calls such as aten_op(quantized=True) to class
    attrs.
    """

    if method_name not in {"aten_op", "exir_op"}:
        return None

    quantized = _call_keyword_bool(call, "quantized", default=False)
    if quantized:
        return f"quantized_{method_name}"
    return method_name


def _resolve_helper_method_call(
    node: ast.Call,
    context: ModuleContext,
    scope: ResolutionScope,
) -> list[str]:
    """Resolve simple test helper method calls returning operator names.

    The important pattern is a parametrized callable that returns a dataclass-like
    object with a ``model`` field. Example::

        @common.parametrize("test_case", test_suite)
        def test_foo(test_case):
            test_case = test_case()
            BackendPipeline(..., aten_op=test_case.aten_op())

    where each suite entry is ``lambda: FooTestCase(Model(), inputs)`` and
    ``FooTestCase.aten_op()`` returns ``getattr(self.model, "aten_op")``.

    """

    if not isinstance(node.func, ast.Attribute):
        return []

    attr = _method_attr_for_call(node.func.attr, node)
    if attr is None:
        return []

    receiver = node.func.value
    values: list[str] = []
    for receiver_candidate in _candidate_exprs(receiver, context, scope):
        if isinstance(receiver_candidate, ast.Call):
            # Direct class helper object: FooTestCase(Foo(), ...).aten_op()
            values.extend(
                _model_attr_from_test_case_call(
                    receiver_candidate, attr, context, scope
                )
            )

            # Direct model instance: Foo().aten_op() is unusual but cheap to
            # support when the class attribute exists.
            class_name = _call_class_name(receiver_candidate)
            if class_name is not None:
                values.extend(
                    _op_texts(
                        _class_attr_value(class_name, attr, context),
                        context,
                        _scope_with_class(scope, class_name),
                    )
                )
        elif isinstance(receiver_candidate, ast.Name):
            values.extend(
                _op_texts(
                    _class_attr_value(receiver_candidate.id, attr, context),
                    context,
                    _scope_with_class(scope, receiver_candidate.id),
                )
            )
    return values


def _op_texts(  # noqa: C901
    node: ast.AST | None,
    context: ModuleContext,
    scope: ResolutionScope | None = None,
    seen: set[tuple[str, str]] | None = None,
) -> list[str]:
    """Return raw operator-name strings represented by a static AST
    expression.
    """

    if node is None:
        return []
    if scope is None:
        scope = ResolutionScope()
    if seen is None:
        seen = set()

    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return [node.value]
        return []

    if isinstance(node, ast.Name):
        name_values: list[str] = []
        for candidate in _resolve_name_expr(node.id, context, scope, seen):
            name_values.extend(_op_texts(candidate, context, scope, seen))
        return name_values

    if isinstance(node, ast.Subscript):
        subscript_values = _op_texts(node.value, context, scope, seen.copy())
        index = node.slice.value if isinstance(node.slice, ast.Constant) else None
        if isinstance(index, int):
            return (
                subscript_values[index : index + 1]
                if index >= 0
                else subscript_values[index:]
            )
        return subscript_values

    if isinstance(node, ast.Attribute):
        # ClassName.aten_op or a local instance such as model.aten_op.
        if isinstance(node.value, ast.Name):
            class_value = _class_attr_value(node.value.id, node.attr, context)
            if class_value is not None:
                return _op_texts(
                    class_value,
                    context,
                    _scope_with_class(scope, node.value.id),
                    seen,
                )

            instance_attribute_values: list[str] = []
            for candidate in _candidate_exprs(node.value, context, scope, seen.copy()):
                if isinstance(candidate, ast.Call):
                    class_name = _call_class_name(candidate)
                    if class_name is not None:
                        instance_attribute_values.extend(
                            _op_texts(
                                _class_attr_value(class_name, node.attr, context),
                                context,
                                _scope_with_class(scope, class_name),
                                seen.copy(),
                            )
                        )
            if instance_attribute_values:
                return instance_attribute_values

        # ClassName().aten_op or local_factory().aten_op
        if isinstance(node.value, ast.Call):
            call_attribute_values: list[str] = []
            for candidate in _candidate_exprs(node.value, context, scope, seen):
                if isinstance(candidate, ast.Call):
                    class_name = _call_class_name(candidate)
                    if class_name is not None:
                        call_attribute_values.extend(
                            _op_texts(
                                _class_attr_value(class_name, node.attr, context),
                                context,
                                _scope_with_class(scope, class_name),
                                seen,
                            )
                        )
            if call_attribute_values:
                return call_attribute_values

        # Preserve direct operator-handle expressions such as
        # torch.ops.aten.neg.default, but still reject unresolved helper
        # expressions such as Neg.aten_op. The normalizer only accepts
        # ATen/Edge namespaces, so it is safe to use as a guard here.
        text = _attribute_name(node)
        return [text] if _normalize_pytorch_op_name(text) is not None else []

    if isinstance(node, ast.Call):
        getattr_values = _resolve_getattr_call(node, context, scope)
        if getattr_values:
            return getattr_values

        helper_values = _resolve_helper_method_call(node, context, scope)
        if helper_values:
            return helper_values

        if _call_name(node.func) == "param" and node.args:
            return _op_texts(node.args[0], context, scope, seen)

        # Calling a parametrized callable, e.g. test_module(), can reveal a
        # lambda body. This is mainly useful when an operator expression itself
        # is wrapped in a helper value.
        candidates = _candidate_exprs(node, context, scope, seen)
        if candidates != [node]:
            candidate_values: list[str] = []
            for candidate in candidates:
                candidate_values.extend(_op_texts(candidate, context, scope, seen))
            return candidate_values

        return []

    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        collection_values: list[str] = []
        for element in node.elts:
            collection_values.extend(_op_texts(element, context, scope, seen))
        return collection_values

    if isinstance(node, ast.Dict):
        dictionary_values: list[str] = []
        for value in _dict_value_candidates(node):
            dictionary_values.extend(_op_texts(value, context, scope, seen))
        return dictionary_values

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _op_texts(node.left, context, scope, seen) + _op_texts(
            node.right, context, scope, seen
        )

    return []


def _resolve_exported_ops(
    node: ast.AST | None,
    context: ModuleContext,
    scope: ResolutionScope | None = None,
    diagnostics: list[str] | None = None,
) -> list[str]:
    ops = []
    for value in _op_texts(node, context, scope):
        normalized = _normalize_pytorch_op_name(
            value, path=context.path, diagnostics=diagnostics
        )
        if normalized is not None:
            ops.append(normalized)
    return sorted(set(ops))


def _is_empty_op_expr(node: ast.AST | None) -> bool:
    return isinstance(node, (ast.List, ast.Tuple, ast.Set)) and not node.elts


def _fallback_exported_ops_for_unattributed_pipeline_call(
    context: ModuleContext,
    scope: ResolutionScope,
) -> list[str]:
    """Infer coverage for op-level tests that pass an empty ATen op list.

    Several op tests intentionally disable operator-distribution assertions in
    the configured backend pipeline by passing empty operator lists even though
    the module still defines a single module-level ``aten_op``/``aten_ops``
    constant. Treat those as backend coverage for --check/debug. Multi-op and
    model tests stay unattributed.

    """

    if "ops" not in context.path.parts:
        return []

    candidates: list[str] = []
    for name in ("aten_op", "aten_ops"):
        assigned = context.assignments.get(name)
        if assigned is not None:
            candidates.extend(_resolve_exported_ops(assigned, context, scope))

    unique = sorted(set(candidates))
    return unique if len(unique) == 1 else []


def _has_xfail_marker(node: ast.AST) -> bool:
    return "xfail" in _expr_text(node).lower()


def _dtype_from_expr(node: ast.AST) -> str | None:
    text = _expr_text(node).lower()
    attr = _attribute_name(node).lower() if isinstance(node, ast.Attribute) else ""
    combined = f"{text} {attr}"
    if "bfloat16" in combined or "bf16" in combined:
        return "BF16"
    if "float16" in combined or "fp16" in combined:
        return "FP16"
    if "float32" in combined or "fp32" in combined:
        return "FP32"
    if "torch.bool" in combined or "dtype=torch.bool" in combined:
        return "BOOL"
    return None


def _dtype_hints_from_data_expr(node: ast.AST, context: ModuleContext) -> set[str]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _dtype_hints_from_data_expr(
            node.left, context
        ) | _dtype_hints_from_data_expr(node.right, context)

    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        collection_dtypes: set[str] = set()
        for element in node.elts:
            collection_dtypes.update(_dtype_hints_from_data_expr(element, context))
        return collection_dtypes

    if isinstance(node, ast.Dict):
        dictionary_dtypes: set[str] = set()
        for value in _dict_value_candidates(node):
            dictionary_dtypes.update(_dtype_hints_from_data_expr(value, context))
        return dictionary_dtypes

    if isinstance(node, ast.Call):
        name = _call_name(node.func)
        if name in context.functions:
            return _dtype_hints_from_helper(context.functions[name])
        return _dtype_hints_from_text(_expr_text(node))

    if isinstance(node, ast.Name):
        assigned = context.assignments.get(node.id)
        if assigned is not None:
            return _dtype_hints_from_data_expr(assigned, context)
        return _dtype_hints_from_text(node.id)

    if isinstance(node, ast.Attribute):
        return _dtype_hints_from_text(_attribute_name(node))

    return _dtype_hints_from_text(_expr_text(node))


def _dtype_hints_from_helper(function: ast.FunctionDef) -> set[str]:
    dtypes: set[str] = set()

    def collect(node: ast.AST) -> None:
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for element in node.elts:
                collect(element)
            return

        if isinstance(node, ast.Call) and _call_name(node.func) == "param":
            # Do not claim support from parameter cases that are explicitly
            # marked xfail, for example BF16 cases that exist to document
            # missing runtime support.
            if _has_xfail_marker(node):
                return
            if node.args:
                collect(node.args[0])
            return

        dtype = _dtype_from_expr(node)
        if dtype is not None:
            dtypes.add(dtype)

    for return_node in [
        node for node in ast.walk(function) if isinstance(node, ast.Return)
    ]:
        if return_node.value is not None:
            collect(return_node.value)
    return dtypes


def _dtype_hints_from_text(text: str) -> set[str]:
    value = text.lower()
    dtypes: set[str] = set()
    if "test_data_bf16" in value or "bfloat16" in value or "bf16" in value:
        dtypes.add("BF16")
    if "test_data_fp16" in value or "float16" in value or "fp16" in value:
        dtypes.add("FP16")
    if "float32" in value or "fp32" in value:
        dtypes.add("FP32")
    if "torch.bool" in value or "dtype=torch.bool" in value:
        dtypes.add("BOOL")
    if re.search(r"(^|\.)test_data($|[^a-z0-9_])", value):
        dtypes.add("FP32")
    return dtypes


def _infer_fp_dtypes(
    function: ast.FunctionDef, call: ast.Call, context: ModuleContext
) -> set[str]:
    dtypes: set[str] = set()

    test_data_expr = _positional_or_keyword(call, 1, "test_data")
    if test_data_expr is not None:
        dtypes.update(_dtype_hints_from_data_expr(test_data_expr, context))

    for decorator in function.decorator_list:
        decorator_text = _expr_text(decorator).lower()
        if "parametrize" not in decorator_text:
            continue
        if isinstance(decorator, ast.Call) and len(decorator.args) >= 2:
            param_name = decorator.args[0]
            if isinstance(param_name, ast.Constant) and isinstance(
                param_name.value, str
            ):
                if "test_data" in param_name.value or "dtype" in param_name.value:
                    dtypes.update(
                        _dtype_hints_from_data_expr(decorator.args[1], context)
                    )
        else:
            dtypes.update(_dtype_hints_from_text(decorator_text))

    # The common test_data dictionaries use torch.randn/torch.ones without an
    # explicit dtype, which means FP32. Keep that as the default FP path.
    if not dtypes:
        dtypes.add("FP32")
    return dtypes


def _function_has_a16w8_quantization(function: ast.FunctionDef, call: ast.Call) -> bool:
    function_text = _expr_text(function)
    call_text = _expr_text(call)
    return (
        "get_symmetric_a16w8_quantization_config" in function_text
        or "a16w8_quantization=True" in call_text
        or "tosa_extensions=['int16']" in call_text
        or 'tosa_extensions=["int16"]' in call_text
    )


def _function_has_a8w4_quantization(function: ast.FunctionDef, call: ast.Call) -> bool:
    function_text = _expr_text(function)
    call_text = _expr_text(call)
    return (
        "get_symmetric_a8w4_quantization_config" in function_text
        or "a8w4_quantization=True" in call_text
    )


def _function_is_skipped_or_xfailed(function: ast.FunctionDef) -> bool:
    for decorator in function.decorator_list:
        text = _expr_text(decorator).lower()
        # Skip function-level xfails. Parameter-level xfails are handled by the
        # dtype helper parser when possible.
        if "xfail" in text and "parametrize" not in text:
            return True
    return False


def _class_context(node: ast.ClassDef) -> ClassContext:
    ctx = ClassContext()

    for stmt in node.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    ctx.assignments[target.id] = stmt.value
                    if target.id not in ctx.fields:
                        ctx.fields.append(target.id)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if stmt.target.id not in ctx.fields:
                ctx.fields.append(stmt.target.id)
            if stmt.value is not None:
                ctx.assignments[stmt.target.id] = stmt.value
        elif isinstance(stmt, ast.FunctionDef):
            ctx.methods[stmt.name] = stmt

    return ctx


def _build_module_context(  # noqa: C901
    path: Path, repo_root: Path
) -> ModuleContext | None:
    try:
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError):
        return None

    assignments: dict[str, ast.AST] = {}
    functions: dict[str, ast.FunctionDef] = {}
    classes: dict[str, ClassContext] = {}
    parents: dict[ast.AST, ast.AST] = {}

    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                assignments[node.target.id] = node.value
        elif isinstance(node, ast.FunctionDef):
            functions[node.name] = node
        elif isinstance(node, ast.ClassDef):
            classes[node.name] = _class_context(node)

    return ModuleContext(
        path=path.relative_to(repo_root),
        source=source,
        tree=tree,
        assignments=assignments,
        functions=functions,
        classes=classes,
        parents=parents,
    )


def _find_enclosing_function(
    node: ast.AST, context: ModuleContext
) -> ast.FunctionDef | None:
    current = node
    while current in context.parents:
        current = context.parents[current]
        if isinstance(current, ast.FunctionDef):
            return current
    return None


def _add_coverage(
    rows: dict[str, PipelineCoverage],
    op: str,
    profile: str,
    stage: str,
    test_label: str,
    function: ast.FunctionDef,
    call: ast.Call,
    context: ModuleContext,
    *,
    classification: str = DIRECT,
    asserted_op: str | None = None,
) -> None:
    row = rows.setdefault(
        op,
        PipelineCoverage(
            exported_op=op,
            pytorch_apis=_pytorch_api_aliases(op),
        ),
    )
    row.tests.add(test_label)
    row.support_profiles.add(profile)
    row.stages.add(stage)
    row.classifications.add(classification)
    row.evidence_records.append(
        CoverageEvidence(
            exported_op=op,
            profile=profile,
            stage=stage,
            classification=classification,
            test=test_label,
            asserted_op=asserted_op or op,
        )
    )

    if profile == "INT":
        if _function_has_a16w8_quantization(function, call):
            row.dtypes.add("INT16")
            row.quantization_modes.add("16x8")
        elif _function_has_a8w4_quantization(function, call):
            row.dtypes.add("INT4")
            row.quantization_modes.add("8x4")
        else:
            row.dtypes.add("INT8")
            row.quantization_modes.add("8x8")
    else:
        row.dtypes.update(_infer_fp_dtypes(function, call, context))


def _removed_pipeline_stages(
    function: ast.FunctionDef, pipeline_var: str | None
) -> set[str]:
    if pipeline_var is None:
        return set()
    removed: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "pop_stage" or not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id != pipeline_var or not node.args:
            continue
        if isinstance(node.args[0], ast.Constant) and isinstance(
            node.args[0].value, str
        ):
            removed.add(node.args[0].value)
    return removed


def _assigned_pipeline_name(call: ast.Call, context: ModuleContext) -> str | None:
    parent = context.parents.get(call)
    if (
        isinstance(parent, ast.Assign)
        and len(parent.targets) == 1
        and isinstance(parent.targets[0], ast.Name)
    ):
        return parent.targets[0].id
    if isinstance(parent, ast.AnnAssign) and isinstance(parent.target, ast.Name):
        return parent.target.id
    return None


def _scan_backend_pipeline_tests(  # noqa: C901
    repo_root: Path,
) -> tuple[dict[str, PipelineCoverage], list[UnresolvedPipelineEvidence], list[str]]:
    rows: dict[str, PipelineCoverage] = {}
    unresolved: list[UnresolvedPipelineEvidence] = []
    normalisation_diagnostics: list[str] = []
    test_dir = repo_root / TEST_ROOT
    if not test_dir.is_dir():
        raise RuntimeError(
            f"Could not locate {BACKEND_NAME} test directory: {test_dir}"
        )

    for path in sorted(test_dir.rglob("*.py")):
        context = _build_module_context(path, repo_root)
        if context is None:
            continue

        for node in ast.walk(context.tree):
            if not isinstance(node, ast.Call) or not _is_backend_pipeline_call(node):
                continue
            function = _find_enclosing_function(node, context)
            if function is None or _function_is_skipped_or_xfailed(function):
                continue

            profile = _pipeline_profile(node)
            test_label = f"{context.path}::{function.name}"
            scope = _scope_for_call(function, node, context)
            aten_expr = _pipeline_operator_expr(
                node, PIPELINE_ATEN_OP_POSITION, PIPELINE_ATEN_OP_KEYWORDS
            )
            exir_expr = _pipeline_operator_expr(
                node, PIPELINE_EXIR_OP_POSITION, PIPELINE_EXIR_OP_KEYWORDS
            )

            diagnostics: list[str] = []
            aten_ops = _resolve_exported_ops(aten_expr, context, scope, diagnostics)
            exir_ops = _resolve_exported_ops(exir_expr, context, scope, diagnostics)
            normalisation_diagnostics.extend(
                f"{test_label}: {message}" for message in diagnostics
            )

            removed = _removed_pipeline_stages(
                function, _assigned_pipeline_name(node, context)
            )
            aten_assertion_removed = bool({"check.aten", "check_count.aten"} & removed)
            exir_assertion_removed = bool({"check.exir", "check_count.exir"} & removed)

            if not aten_assertion_removed:
                for op in aten_ops:
                    _add_coverage(
                        rows,
                        op,
                        profile,
                        SOURCE_ATEN,
                        test_label,
                        function,
                        node,
                        context,
                        asserted_op=op,
                    )
            if not exir_assertion_removed:
                for op in exir_ops:
                    _add_coverage(
                        rows,
                        op,
                        profile,
                        EDGE_IR,
                        test_label,
                        function,
                        node,
                        context,
                        asserted_op=op,
                    )

            explicit = EXPLICIT_BACKEND_COVERAGE.get(
                (str(context.path), function.name), {}
            )
            for explicit_op in explicit.get(profile, set()):
                _add_coverage(
                    rows,
                    explicit_op,
                    profile,
                    RUNTIME_ONLY,
                    test_label,
                    function,
                    node,
                    context,
                    classification=EXPLICIT,
                )

            if not aten_ops and not exir_ops and not explicit.get(profile):
                fallback: list[str] = []
                if _is_empty_op_expr(aten_expr):
                    fallback = _fallback_exported_ops_for_unattributed_pipeline_call(
                        context, scope
                    )
                for op in fallback:
                    _add_coverage(
                        rows,
                        op,
                        profile,
                        RUNTIME_ONLY,
                        test_label,
                        function,
                        node,
                        context,
                        classification=INFERRED,
                    )
                if not fallback:
                    unresolved.append(
                        UnresolvedPipelineEvidence(
                            path=context.path,
                            function=function.name,
                            profile=profile,
                            aten_expression=_expr_text(aten_expr),
                            exir_expression=_expr_text(exir_expr),
                            reason="no statically attributable ATen or Edge operator",
                        )
                    )

    return rows, unresolved, sorted(set(normalisation_diagnostics))


def _public_rows_from_exact_rows(
    exact_rows: Mapping[str, PipelineCoverage],
) -> list[PublicCoverage]:
    public_rows: dict[str, PublicCoverage] = {}
    for exact in exact_rows.values():
        if not exact.pytorch_apis:
            continue

        key = _api_key(exact.pytorch_apis)
        row = public_rows.setdefault(
            key,
            PublicCoverage(pytorch_apis=exact.pytorch_apis),
        )
        row.support_profiles.update(exact.support_profiles)
        row.dtypes.update(exact.dtypes)
        row.quantization_modes.update(exact.quantization_modes)

    return sorted(public_rows.values(), key=lambda row: _api_sort_key(row.pytorch_apis))


def _profiles_for_checker(
    checker: type, backend_tosa_spec: TosaSpecificationLike
) -> set[str]:
    """Best-effort profile extraction for registered support checks.

    Some checkers are unconditional over all profiles; others declare a narrower
    ``tosa_specs`` class attribute. This cannot model dtype predicates inside
    ``is_node_tosa_supported`` but avoids assigning a checker to a profile that
    its own declared spec list excludes.

    """

    profiles: set[str] = set()
    tosa_specs = getattr(checker, "tosa_specs", None)

    if tosa_specs is None:
        if backend_tosa_spec.support_float():
            profiles.add("FP")
        if backend_tosa_spec.support_integer():
            profiles.add("INT")
        return profiles

    try:
        specs = list(tosa_specs)
    except TypeError:
        specs = [tosa_specs]

    for raw_spec in specs:
        spec = cast(TosaSpecificationLike, raw_spec)
        try:
            if spec.support_float() and backend_tosa_spec.support_float():
                profiles.add("FP")
            if spec.support_integer() and backend_tosa_spec.support_integer():
                profiles.add("INT")
        except Exception as error:
            logger.debug("Skipping unsupported TOSA spec %r: %s", spec, error)
            continue

    return profiles


def _collect_backend_supported_ops(  # noqa: C901
    repo_root: Path,
) -> dict[str, SupportedOperatorEvidence]:
    """Collect exported ATen ops that backend registries say should be
    supported.
    """

    _ensure_repo_importable(repo_root)

    # Importing the package registers all @register_tosa_support_check classes.
    import executorch.backends.arm.operator_support  # noqa: F401

    from executorch.backends.arm.operator_support import (
        tosa_profile_supported_op_lists as profile_op_lists,
        tosa_supported_operators,
    )
    from executorch.backends.arm.tosa import TosaSpecification

    tosa_spec = TosaSpecification.create_from_string(BACKEND_TOSA_SPEC)
    expected: dict[str, SupportedOperatorEvidence] = {}

    def add(target: object, profile: str, evidence: str) -> None:
        exported_op = _canonical_pytorch_op_from_target(target)
        if exported_op is None:
            return
        row = expected.setdefault(
            exported_op,
            SupportedOperatorEvidence(
                exported_op=exported_op,
                pytorch_apis=_pytorch_api_aliases(exported_op),
            ),
        )
        row.support_profiles.add(profile)
        row.evidence.add(evidence)

    if tosa_spec.support_float():
        for target in profile_op_lists.TOSA_PRO_FP_SupportList:
            add(target, "FP", "TOSA_PRO_FP_SupportList")

    if tosa_spec.support_integer():
        for target in profile_op_lists.TOSA_PRO_INT_SupportList:
            add(target, "INT", "TOSA_PRO_INT_SupportList")

    for checker in tosa_supported_operators.get_registered_tosa_support_checks(
        tosa_spec
    ):
        checker_evidence = f"registered support check `{checker.__name__}`"
        for target in getattr(checker, "targets", ()):  # type: ignore[attr-defined]
            for profile in _profiles_for_checker(checker, tosa_spec):
                add(target, profile, checker_evidence)

    # Lowering visitors are not the source of partitioner support, but they are
    # useful evidence when the exported op name matches a registered visitor
    # target directly.
    try:
        from executorch.backends.arm.operators.node_visitor import get_node_visitors

        for target, visitor in get_node_visitors(tosa_spec).items():
            exported_op = _normalize_pytorch_op_name(target)
            if exported_op is None or exported_op not in expected:
                continue
            expected[exported_op].evidence.add(
                f"lowering visitor `{visitor.__class__.__name__}`"
            )
    except Exception as error:
        # Keep --check focused on missing test diagnostics even if optional
        # lowering visitor imports fail in a reduced environment.
        logger.debug("Could not collect optional lowering visitors: %s", error)

    return expected


def _format_markdown_table_row(cells: Sequence[str]) -> str:
    # Serializes a list of cell values into a Markdown table row.
    # It escapes literal | characters inside cells so they aren’t
    # interpreted as column separators,
    # then joins the cells using Markdown’s | delimiter.
    escaped_cells = (cell.replace("|", r"\|") for cell in cells)
    return "| " + " | ".join(escaped_cells) + " |"


def generate_markdown(repo_root: Path, *, debug: bool = False) -> str:
    exact_rows, _unresolved, _diagnostics = _scan_backend_pipeline_tests(repo_root)
    command = GENERATOR_COMMAND
    if debug:
        command += " --debug"

    lines = [
        f"# {PAGE_TITLE}",
        "",
        f"<!-- DO NOT EDIT: generated by `{command}`. -->",
        "",
        PAGE_DESCRIPTION,
        "",
        "`8x8` means 8-bit activations and 8-bit weights. `16x8` means "
        "16-bit activations and 8-bit weights. `8x4` means "
        "8-bit activations and 4-bit weights.",
        "",
    ]

    if debug:
        sorted_rows = sorted(
            exact_rows.values(),
            key=lambda row: (_api_sort_key(row.pytorch_apis), row.exported_op),
        )
        lines.extend(
            [
                f"Total tested exported operators: **{len(sorted_rows)}**.",
                "",
                "```{note}",
                MARKDOWN_DEBUG_NOTE,
                "```",
                "",
                "| PyTorch API | Exported operator | Support profile | DType | Quantization mode | Test |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for exact_row in sorted_rows:
            cells = [
                _format_api_items(exact_row.pytorch_apis),
                f"`{exact_row.exported_op}`",
                _format_items(exact_row.support_profiles, SUPPORT_PROFILE_ORDER),
                _format_backtick_items(exact_row.dtypes, DTYPE_ORDER),
                _format_items(exact_row.quantization_modes, QUANTIZATION_MODE_ORDER),
                _format_test_items(exact_row.tests),
            ]
            lines.append(_format_markdown_table_row(cells))
    else:
        public_rows = _public_rows_from_exact_rows(exact_rows)
        lines.extend(
            [
                f"Total supported PyTorch APIs: **{len(public_rows)}**.",
                "",
                "| PyTorch API | Support profile | DType | Quantization mode |",
                "| --- | --- | --- | --- |",
            ]
        )
        for public_row in public_rows:
            cells = [
                _format_api_items(public_row.pytorch_apis),
                _format_items(public_row.support_profiles, SUPPORT_PROFILE_ORDER),
                _format_backtick_items(public_row.dtypes, DTYPE_ORDER),
                _format_items(public_row.quantization_modes, QUANTIZATION_MODE_ORDER),
            ]
            lines.append(_format_markdown_table_row(cells))
    return "\n".join(lines).rstrip() + "\n"


def generate_html(repo_root: Path, *, debug: bool = False) -> str:
    """Generate a standalone HTML version of the operator-support page."""

    exact_rows, _unresolved, _diagnostics = _scan_backend_pipeline_tests(repo_root)
    command = GENERATOR_COMMAND
    if debug:
        command += " --debug"
    command += " --html"

    lines = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8" />',
        '  <meta name="viewport" content="width=device-width, initial-scale=1" />',
        f"  <title>{html.escape(PAGE_TITLE)}</title>",
        "  <style>",
        "    :root { color-scheme: light dark; }",
        "    body {",
        "      font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;",
        "      line-height: 1.5;",
        "      margin: 0 auto;",
        "      max-width: 1600px;",
        "      padding: 2rem;",
        "    }",
        "    h1 { line-height: 1.2; }",
        "    .generated { opacity: 0.75; }",
        "    .note {",
        "      border-left: 0.3rem solid currentColor;",
        "      margin: 1.5rem 0;",
        "      padding: 0.75rem 1rem;",
        "    }",
        "    .table-container { overflow-x: auto; }",
        "    table { border-collapse: collapse; width: 100%; }",
        "    th, td { border: 1px solid #8888; padding: 0.55rem; text-align: left; vertical-align: top; }",
        "    th { background: #8882; position: sticky; top: 0; }",
        "    tbody tr:nth-child(even) { background: #8881; }",
        "    code { white-space: nowrap; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <main>",
        f"    <h1>{html.escape(PAGE_TITLE)}</h1>",
        f'    <p class="generated">Generated by {_html_code(command)}.</p>',
        f"    <p>{html.escape(PAGE_DESCRIPTION)}</p>",
        "    <p><code>8x8</code> means 8-bit activations and 8-bit weights. <code>16x8</code> means 16-bit activations and 8-bit weights.</p>",
    ]

    if debug:
        sorted_rows = sorted(
            exact_rows.values(),
            key=lambda row: (_api_sort_key(row.pytorch_apis), row.exported_op),
        )
        lines.extend(
            [
                f"    <p>Total tested exported operators: <strong>{len(sorted_rows)}</strong>.</p>",
                f'    <div class="note">{html.escape(HTML_DEBUG_NOTE)}</div>',
                '    <div class="table-container">',
                "      <table>",
                "        <thead>",
                "          <tr>",
                '            <th scope="col">PyTorch API</th>',
                '            <th scope="col">Exported operator</th>',
                '            <th scope="col">Support profile</th>',
                '            <th scope="col">DType</th>',
                '            <th scope="col">Quantization mode</th>',
                '            <th scope="col">Test</th>',
                "          </tr>",
                "        </thead>",
                "        <tbody>",
            ]
        )
        for exact_row in sorted_rows:
            cells = [
                _format_html_api_items(exact_row.pytorch_apis),
                _html_code(exact_row.exported_op),
                _format_html_items(exact_row.support_profiles, SUPPORT_PROFILE_ORDER),
                _format_html_code_items(exact_row.dtypes, DTYPE_ORDER),
                _format_html_items(
                    exact_row.quantization_modes, QUANTIZATION_MODE_ORDER
                ),
                _format_html_test_items(exact_row.tests),
            ]
            lines.append(
                "          <tr>"
                + "".join(f"<td>{cell}</td>" for cell in cells)
                + "</tr>"
            )
    else:
        public_rows = _public_rows_from_exact_rows(exact_rows)
        lines.extend(
            [
                f"    <p>Total supported PyTorch APIs: <strong>{len(public_rows)}</strong>.</p>",
                '    <div class="table-container">',
                "      <table>",
                "        <thead>",
                "          <tr>",
                '            <th scope="col">PyTorch API</th>',
                '            <th scope="col">Support profile</th>',
                '            <th scope="col">DType</th>',
                '            <th scope="col">Quantization mode</th>',
                "          </tr>",
                "        </thead>",
                "        <tbody>",
            ]
        )
        for public_row in public_rows:
            cells = [
                _format_html_api_items(public_row.pytorch_apis),
                _format_html_items(public_row.support_profiles, SUPPORT_PROFILE_ORDER),
                _format_html_code_items(public_row.dtypes, DTYPE_ORDER),
                _format_html_items(
                    public_row.quantization_modes, QUANTIZATION_MODE_ORDER
                ),
            ]
            lines.append(
                "          <tr>"
                + "".join(f"<td>{cell}</td>" for cell in cells)
                + "</tr>"
            )

    lines.extend(
        [
            "        </tbody>",
            "      </table>",
            "    </div>",
            "  </main>",
            "</body>",
            "</html>",
        ]
    )
    return "\n".join(lines) + "\n"


def _coverage_aliases(expected_op: str, profile: str) -> set[str]:
    return {
        expected_op,
        *STAGE_EQUIVALENT_OPS.get(expected_op, set()),
        *PROFILE_STAGE_EQUIVALENT_OPS.get((expected_op, profile), set()),
    }


def _matching_evidence(
    tested: Mapping[str, PipelineCoverage],
    expected_op: str,
    profile: str,
) -> list[CoverageEvidence]:
    records: list[CoverageEvidence] = []
    for alias in _coverage_aliases(expected_op, profile):
        coverage = tested.get(alias)
        if coverage is None:
            continue
        for record in coverage.evidence_records:
            if record.profile != profile:
                continue
            classification = record.classification
            if alias != expected_op and classification == DIRECT:
                classification = STAGE_EQUIVALENT
            records.append(
                CoverageEvidence(
                    exported_op=expected_op,
                    profile=profile,
                    stage=record.stage,
                    classification=classification,
                    test=record.test,
                    asserted_op=alias,
                )
            )
    return records


def _validate_configuration(repo_root: Path) -> list[str]:
    errors: list[str] = []
    valid_profiles = set(SUPPORT_PROFILE_ORDER)
    for (op, profile), aliases in PROFILE_STAGE_EQUIVALENT_OPS.items():
        if profile not in valid_profiles:
            errors.append(f"invalid profile {profile!r} for {op}")
        for candidate in {op, *aliases}:
            if _normalize_pytorch_op_name(candidate) != candidate:
                errors.append(f"non-canonical equivalence operator: {candidate}")
    overlap = TRANSFORM_ONLY_OPS & DECOMPOSED_OPS
    if overlap:
        errors.append(
            f"operators configured as both transform-only and decomposed: {sorted(overlap)}"
        )
    for (path, function), profiles in EXPLICIT_BACKEND_COVERAGE.items():
        full_path = repo_root / path
        if not full_path.is_file():
            errors.append(f"explicit coverage path does not exist: {path}")
            continue
        context = _build_module_context(full_path, repo_root)
        if context is not None and function not in context.functions:
            errors.append(
                f"explicit coverage function does not exist: {path}::{function}"
            )
        for profile in profiles:
            if profile not in valid_profiles:
                errors.append(
                    f"invalid explicit coverage profile {profile!r}: {path}::{function}"
                )
    return errors


def _print_unresolved(unresolved: Sequence[UnresolvedPipelineEvidence]) -> None:
    if not unresolved:
        return
    print(f"Unresolved {BACKEND_PIPELINE_LABEL} attribution:")
    print()
    print("| Test | Profile | ATen expression | Edge expression | Reason |")
    print("| --- | --- | --- | --- | --- |")
    for item in unresolved:
        print(
            f"| `{item.path}::{item.function}` | {item.profile} | "
            f"`{item.aten_expression or '-'}` | `{item.exir_expression or '-'}` | "
            f"{item.reason} |"
        )
    print()


def explain_operator(repo_root: Path, requested_op: str) -> int:
    tested, unresolved, diagnostics = _scan_backend_pipeline_tests(repo_root)
    expected = _collect_backend_supported_ops(repo_root)
    op = _normalize_pytorch_op_name(requested_op) or requested_op
    row = expected.get(op)
    if row is None:
        print(f"Operator is not present in the {BACKEND_NAME} support registry: {op}")
        return 1
    print(f"Operator: {op}")
    print(
        f"Expected profiles: {_format_items(row.support_profiles, SUPPORT_PROFILE_ORDER)}"
    )
    print()
    for profile in _sort_items(row.support_profiles, SUPPORT_PROFILE_ORDER):
        print(f"{profile}:")
        if op in TRANSFORM_ONLY_OPS:
            print("  transform-only exemption")
            continue
        records = _matching_evidence(tested, op, profile)
        if not records:
            print("  missing")
            continue
        for record in sorted(
            records, key=lambda r: (r.classification, r.test, r.asserted_op or "")
        ):
            print(
                f"  {record.classification}; {record.stage}; "
                f"satisfied by {record.asserted_op}; {record.test}"
            )
    if diagnostics:
        print()
        print("Normalisation diagnostics:")
        for diagnostic in diagnostics:
            if op.rsplit(".", 1)[0].split(".")[-1] in diagnostic:
                print(f"  {diagnostic}")
    relevant_unresolved = [
        item
        for item in unresolved
        if op.split(".")[-2] in (item.aten_expression + item.exir_expression)
    ]
    if relevant_unresolved:
        print()
        _print_unresolved(relevant_unresolved)
    return 0


def run_check(repo_root: Path, *, strict_ast: bool = False) -> int:  # noqa: C901
    config_errors = _validate_configuration(repo_root)
    if config_errors:
        print(f"Invalid {BACKEND_NAME} support-check configuration:")
        for error in config_errors:
            print(f"- {error}")
        return 2

    tested, unresolved, diagnostics = _scan_backend_pipeline_tests(repo_root)
    expected = _collect_backend_supported_ops(repo_root)

    missing_cells: list[tuple[SupportedOperatorEvidence, str]] = []
    classified: list[tuple[str, str, str, str, str]] = []

    for op, registry_evidence in expected.items():
        for profile in _sort_items(
            registry_evidence.support_profiles, SUPPORT_PROFILE_ORDER
        ):
            if op in TRANSFORM_ONLY_OPS:
                classified.append((op, profile, TRANSFORM_ONLY, "exempt", "-"))
                continue
            records = _matching_evidence(tested, op, profile)
            if not records:
                missing_cells.append((registry_evidence, profile))
                continue
            best = sorted(
                records,
                key=lambda record: (
                    0 if record.classification == DIRECT else 1,
                    record.test,
                ),
            )[0]
            if best.classification != DIRECT or op in DECOMPOSED_OPS:
                classification = (
                    DECOMPOSED if op in DECOMPOSED_OPS else best.classification
                )
                classified.append(
                    (
                        op,
                        profile,
                        classification,
                        best.asserted_op or op,
                        best.test,
                    )
                )

    if classified:
        print("Classified non-direct coverage:")
        print()
        print(
            "| Exported operator | Expected profile | Classification | Satisfied by | Evidence test |"
        )
        print("| --- | --- | --- | --- | --- |")
        for op, profile, classification, satisfied_by, test in sorted(classified):
            sat = f"`{satisfied_by}`" if satisfied_by != "exempt" else "exempt"
            test_cell = f"`{test}`" if test != "-" else "-"
            print(f"| `{op}` | {profile} | {classification} | {sat} | {test_cell} |")
        print()

    if unresolved:
        _print_unresolved(unresolved)
    if diagnostics:
        print("AST normalisation diagnostics:")
        for diagnostic in diagnostics:
            print(f"- {diagnostic}")
        print()

    if not missing_cells:
        print(
            f"All backend-supported exported ATen operator/profile pairs have {BACKEND_PIPELINE_LABEL} coverage."
        )
        return 1 if strict_ast and unresolved else 0

    print(
        f"The following backend-supported exported ATen operator/profile pairs are missing {BACKEND_PIPELINE_LABEL} coverage:"
    )
    print()
    print(
        "| PyTorch API | Exported operator | Missing profile | Expected profile | Evidence |"
    )
    print("| --- | --- | --- | --- | --- |")
    for row, profile in sorted(
        missing_cells,
        key=lambda item: (
            _api_sort_key(item[0].pytorch_apis),
            item[0].exported_op,
            item[1],
        ),
    ):
        print(
            "| "
            f"{_format_api_items(row.pytorch_apis)} | "
            f"`{row.exported_op}` | {profile} | "
            f"{_format_items(row.support_profiles, SUPPORT_PROFILE_ORDER)} | "
            f"{_format_items(row.evidence)} |"
        )

    missing_ops = {row.exported_op for row, _profile in missing_cells}
    print()
    print(
        f"Missing {len(missing_cells)} profile cells across {len(missing_ops)} "
        f"of {len(expected)} backend-supported exported ATen operators."
    )
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=CLI_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Path to the ExecuTorch repository root. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Markdown output path. Defaults to {DEFAULT_OUTPUT} under repo root.",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help=(
            "Also write a standalone HTML page next to the Markdown output, "
            "using the same file name with a .html suffix."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include maintainer-facing Exported operator and Test columns.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            f"Do not write the page. Compare exact {BACKEND_PIPELINE_LABEL} "
            "exported ATen coverage against backend support registries and "
            f"print supported ops missing {BACKEND_NAME} tests."
        ),
    )
    parser.add_argument(
        "--strict-ast",
        action="store_true",
        help=(
            f"Fail --check when any {BACKEND_PIPELINE_LABEL} call cannot be "
            "statically attributed."
        ),
    )
    parser.add_argument(
        "--explain",
        metavar="EXPORTED_OP",
        help="Explain how one exported ATen operator is covered per profile.",
    )
    args = parser.parse_args(argv)

    root = _repo_root(args.repo_root)

    if args.explain:
        return explain_operator(root, args.explain)
    if args.check:
        return run_check(root, strict_ast=args.strict_ast)

    output = args.output or root / DEFAULT_OUTPUT
    if not output.is_absolute():
        output = root / output

    markdown = generate_markdown(root, debug=args.debug)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {output}")

    if args.html:
        html_output = output.with_suffix(".html")
        html_page = generate_html(root, debug=args.debug)
        html_output.write_text(html_page, encoding="utf-8")
        print(f"Wrote {html_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
