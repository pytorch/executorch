# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from executorch.backends.arm.test.benchmarks.sdpa_baseline import (
    build_custom_sdpa_module,
    build_decode_case,
    check_case_correctness,
    export_and_decompose,
    SdpaDecodeConfig,
)


def test_sdpa_baseline_decode_case_matches_decomposed_graph() -> None:
    case = build_decode_case(128, SdpaDecodeConfig())
    check_case_correctness(case, atol=1e-5, rtol=1e-5)


def test_sdpa_baseline_decomposition_removes_sdpa_op() -> None:
    case = build_decode_case(512, SdpaDecodeConfig())
    graph_module, _ = export_and_decompose(case)
    assert not any(
        node.target == torch.ops.aten.scaled_dot_product_attention.default
        for node in graph_module.graph.nodes
        if node.op == "call_function"
    )


def test_sdpa_baseline_custom_op_matches_eager_decode_case() -> None:
    case = build_decode_case(512, SdpaDecodeConfig())
    eager_output = case.module(*case.inputs)
    custom_output = build_custom_sdpa_module()(*case.inputs)
    torch.testing.assert_close(custom_output, eager_output, atol=1e-5, rtol=1e-5)


def test_sdpa_baseline_rejects_non_decode_query_length() -> None:
    with pytest.raises(ValueError, match="single-token decode"):
        build_decode_case(128, SdpaDecodeConfig(query_len=2))
