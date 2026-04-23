# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm._passes import InsertDataLayoutCastsPass
from executorch.backends.arm.test.tester.test_pipeline import PassPipeline
from executorch.backends.test.harness.stages import StageType
from executorch.exir.dialects._ops import ops as exir_ops


def _collect_cast_dtypes(
    pipeline: PassPipeline[tuple[torch.Tensor, ...]],
) -> list[torch.dtype]:
    exported_program = pipeline.tester.get_artifact(
        StageType.RUN_PASSES
    ).exported_program()
    graph_module = exported_program.graph_module

    cast_dtypes: list[torch.dtype] = []
    for node in graph_module.graph.nodes:
        if (
            node.op == "call_function"
            and node.target == exir_ops.edge.dim_order_ops._to_dim_order_copy.default
            and "dtype" in node.kwargs
        ):
            cast_dtypes.append(node.kwargs["dtype"])
    return cast_dtypes


class ViewModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(2, 2)


class CatModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, y], dim=1)


def test_insert_data_layout_casts_no_target_view_fp_profile_inserts_casts() -> None:
    test_data = (torch.arange(4, dtype=torch.int32).reshape(1, 4),)

    pipeline = PassPipeline[tuple[torch.Tensor, ...]](
        ViewModule(),
        test_data,
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
            "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 2,
        },
        pass_list=[InsertDataLayoutCastsPass],
    )
    pipeline.run()

    cast_dtypes = _collect_cast_dtypes(pipeline)
    assert cast_dtypes.count(torch.float32) == 1
    assert cast_dtypes.count(torch.int32) == 1


def test_insert_data_layout_casts_no_target_view_fp_profile_skips_supported_dtype() -> (
    None
):
    test_data = (torch.randn(1, 4, dtype=torch.float32),)

    pipeline = PassPipeline[tuple[torch.Tensor]](
        ViewModule(),
        test_data,
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
        },
        ops_not_after_pass=[
            "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default",
        ],
        pass_list=[InsertDataLayoutCastsPass],
    )
    pipeline.run()


def test_insert_data_layout_casts_no_target_cat_fp_profile_inserts_casts() -> None:
    test_data = (
        torch.arange(4, dtype=torch.int32).reshape(1, 4),
        torch.arange(4, dtype=torch.int32).reshape(1, 4),
    )

    pipeline = PassPipeline[tuple[torch.Tensor, ...]](
        CatModule(),
        test_data,
        quantize=False,
        ops_before_pass={
            "executorch_exir_dialects_edge__ops_aten_cat_default": 1,
        },
        ops_after_pass={
            "executorch_exir_dialects_edge__ops_aten_cat_default": 1,
            "executorch_exir_dialects_edge__ops_dim_order_ops__to_dim_order_copy_default": 3,
        },
        pass_list=[InsertDataLayoutCastsPass],
    )
    pipeline.run()

    cast_dtypes = _collect_cast_dtypes(pipeline)
    assert cast_dtypes.count(torch.float32) == 2
    assert cast_dtypes.count(torch.int32) == 1
