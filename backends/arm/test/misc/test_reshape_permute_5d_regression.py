# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import torch
import torch.nn as nn
from executorch.backends.arm.quantizer import TOSAQuantizer
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_quantization_config as get_arm_symmetric_qconfig,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class ReshapePermuteVariantModel(torch.nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        self.variant = variant
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.block(x)
        if self.variant == "baseline_5d":
            return (
                x1.permute(0, 2, 3, 1).reshape(1, 1, 49, 64, 256).permute(0, 1, 3, 2, 4)
            )
        if self.variant == "workaround_4d_unsqueeze":
            out = x1.permute(0, 2, 3, 1).reshape(1, 49, 64, 256).permute(0, 2, 1, 3)
            return out.unsqueeze(1)
        if self.variant == "singleton_mid_5d":
            return (
                x1.permute(0, 2, 3, 1).reshape(1, 49, 1, 64, 256).permute(0, 2, 4, 1, 3)
            )
        if self.variant == "negative_dims_5d":
            return (
                x1.permute(0, 2, 3, 1)
                .reshape(1, 1, 49, 64, 256)
                .permute(0, 1, -1, -2, -3)
            )
        if self.variant == "nonsingleton_5d":
            return (
                x1.permute(0, 2, 3, 1).reshape(1, 7, 8, 112, 128).permute(0, 2, 1, 4, 3)
            )
        if self.variant == "rank6_singleton":
            return (
                x1.permute(0, 2, 3, 1)
                .reshape(1, 1, 7, 8, 112, 128)
                .permute(0, 1, 3, 2, 5, 4)
            )
        if self.variant == "chained_reshape_permute":
            out = (
                x1.permute(0, 2, 3, 1).reshape(1, 1, 49, 64, 256).permute(0, 1, 3, 2, 4)
            )
            return out.reshape(1, 1, 64, 49, 16, 16).permute(0, 1, 2, 4, 5, 3)
        raise AssertionError(f"Unsupported variant: {self.variant}")


def _run_model(model: torch.nn.Module, out_dir: str) -> Path:
    device = "cpu"
    float_model = model.eval().to(device)

    compile_spec = TosaCompileSpec("TOSA-1.0+INT+int16+int4+cf")
    compile_spec.dump_intermediate_artifacts_to(out_dir).dump_debug_info(
        TosaCompileSpec.DebugMode.JSON
    )

    quantizer = TOSAQuantizer(compile_spec)
    partitioner = TOSAPartitioner(compile_spec)
    quantizer.set_global(get_arm_symmetric_qconfig(is_per_channel=True))

    sample = torch.randn(1, 3, 224, 224).to(device)
    exported_program = torch.export.export(float_model, (sample,))
    graph_module = exported_program.module(check_guards=False)
    prepared = prepare_pt2e(graph_module, quantizer)
    with torch.no_grad():
        prepared(sample)
    quantized_graph_module = convert_pt2e(prepared.to("cpu"), fold_quantize=True)
    quantized_exported_program = torch.export.export(
        quantized_graph_module,
        (torch.randn(1, 3, 224, 224).to(device),),
    )

    to_edge_transform_and_lower(
        quantized_exported_program,
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    tosa_files = sorted(Path(out_dir).glob("*.tosa"))
    assert tosa_files, f"No TOSA artifacts found in {out_dir}"
    return tosa_files[0]


def _assert_all_transposes_shape_consistent(tosa_path: Path) -> None:
    import tosa.Op as Op  # type: ignore[import-not-found,import-untyped]
    from tosa.TosaGraph import (  # type: ignore[import-not-found,import-untyped]
        TosaGraph,
    )
    from tosa.TransposeAttribute import (  # type: ignore[import-not-found,import-untyped]
        TransposeAttribute,
    )

    graph = TosaGraph.GetRootAs(tosa_path.read_bytes(), 0)
    block = graph.Regions(0).Blocks(0)

    shape_by_name = {
        block.Tensors(i).Name().decode(): list(block.Tensors(i).ShapeAsNumpy())
        for i in range(block.TensorsLength())
    }

    op_enum = Op.Op()
    op_value_to_name = {
        getattr(op_enum, name): name for name in dir(op_enum) if name.isupper()
    }

    for i in range(block.OperatorsLength()):
        op = block.Operators(i)
        if op_value_to_name.get(op.Op()) != "TRANSPOSE":
            continue

        inputs = [op.Inputs(j).decode() for j in range(op.InputsLength())]
        outputs = [op.Outputs(j).decode() for j in range(op.OutputsLength())]
        assert len(inputs) == 1 and len(outputs) == 1, (
            f"Unexpected TRANSPOSE arity at op #{i}: "
            f"{len(inputs)} inputs, {len(outputs)} outputs"
        )

        attr_tbl = op.Attribute()
        transpose_attr = TransposeAttribute()
        transpose_attr.Init(attr_tbl.Bytes, attr_tbl.Pos)
        perms = list(transpose_attr.PermsAsNumpy())

        in_shape = shape_by_name[inputs[0]]
        out_shape = shape_by_name[outputs[0]]
        expected_out_shape = [in_shape[perm] for perm in perms]

        assert expected_out_shape == out_shape, (
            f"Invalid TRANSPOSE at op #{i}: perms={perms}, "
            f"in_shape={in_shape}, out_shape={out_shape}, "
            f"expected_out_shape={expected_out_shape}"
        )


@common.parametrize(
    "model_and_name",
    {
        "reshape_permute_5d": ReshapePermuteVariantModel("baseline_5d"),
        "reshape_permute_unsqueeze_4d": ReshapePermuteVariantModel(
            "workaround_4d_unsqueeze"
        ),
        "reshape_permute_singleton_mid_5d": ReshapePermuteVariantModel(
            "singleton_mid_5d"
        ),
        "reshape_permute_negative_dims_5d": ReshapePermuteVariantModel(
            "negative_dims_5d"
        ),
        "reshape_permute_nonsingleton_5d": ReshapePermuteVariantModel(
            "nonsingleton_5d"
        ),
        "reshape_permute_rank6_singleton": ReshapePermuteVariantModel(
            "rank6_singleton"
        ),
        "reshape_permute_chained": ReshapePermuteVariantModel(
            "chained_reshape_permute"
        ),
    },
)
def test_reshape_permute_chains_tosa_INT_transpose_shapes_valid(
    model_and_name: torch.nn.Module, tmp_path
):
    out_dir = tmp_path / "reshape_permute_regression"
    out_dir.mkdir(parents=True, exist_ok=True)
    tosa_path = _run_model(model_and_name, str(out_dir))
    assert tosa_path.exists(), f"Missing TOSA dump: {tosa_path}"
    _assert_all_transposes_shape_consistent(tosa_path)
