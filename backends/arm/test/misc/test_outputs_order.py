# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# pyre-unsafe
import tempfile
from pathlib import Path

import pytest
import torch
from executorch.backends.arm.arm_backend import ArmCompileSpecBuilder
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.tosa_partitioner import TOSAPartitioner
from executorch.backends.arm.tosa_specification import TosaSpecification
from executorch.exir import to_edge_transform_and_lower
from torch import nn
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from tosa import TosaGraph


class Network(nn.Module):
    def __init__(self, batch_norm=False):
        super().__init__()
        self.conv2d_0 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8) if batch_norm else nn.Identity(),
            nn.ReLU(),
        )
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8) if batch_norm else nn.Identity(),
            nn.ReLU(),
        )
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8) if batch_norm else nn.Identity(),
            nn.ReLU(),
        )
        self.out_0 = nn.Sequential(nn.Conv2d(8, 1, 3, padding=1, bias=False), nn.ReLU())
        self.out_1 = nn.Sequential(nn.Conv2d(8, 2, 3, padding=1, bias=False), nn.ReLU())
        self.out_2 = nn.Sequential(nn.Conv2d(8, 3, 3, padding=1, bias=False), nn.ReLU())

    def forward(self, x):
        x = self.conv2d_0(x)
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        out0 = self.out_0(x)
        out1 = self.out_1(x)
        out2 = self.out_2(x)
        return out0, out1, out2


def _read_tosa_outputs(tosa_path: Path):
    # Find output tensor names in order and return shapes
    buf = tosa_path.read_bytes()
    buf_arr = bytearray(buf)
    graph = TosaGraph.TosaGraph.GetRootAsTosaGraph(buf_arr, 0)
    region = graph.Regions(0)
    block = region.Blocks(0)
    # Build a dict name - tensor‑shape
    tensors = {}
    for i in range(block.TensorsLength()):
        t = block.Tensors(i)
        name = t.Name().decode()
        # NHWC
        shape = [t.Shape(j) for j in range(t.ShapeLength())]
        tensors[name] = shape
    shapes = []
    for i in range(block.OutputsLength()):
        out_name = block.Outputs(i).decode()
        shapes.append(tensors[out_name])
    return shapes


@pytest.mark.parametrize("batch_size", [1, 4])
def test_network_output_order_and_restore(tmp_path, batch_size):
    model = Network(batch_norm=True).eval()
    # Prepare spec
    spec = TosaSpecification.create_from_string("TOSA-1.0+INT")
    compile_spec = ArmCompileSpecBuilder().tosa_compile_spec(tosa_spec=spec).build()
    # Setup quantizer
    quantizer = TOSAQuantizer(compile_spec)
    quantizer.set_global(
        get_symmetric_quantization_config(is_qat=True, is_per_channel=False)
    )
    # Trace the model
    dummy = torch.randn(batch_size, 1, 28, 28)
    fx_mod = torch.export.export_for_training(model, (dummy,)).module()
    model = prepare_pt2e(fx_mod, quantizer)
    model(dummy)
    model = convert_pt2e(model)
    # Export to aten dialect
    aten_gm = torch.export.export(model, args=(dummy,), strict=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        art_dir = Path(tmpdir)
        part = TOSAPartitioner(
            ArmCompileSpecBuilder()
            .tosa_compile_spec(spec)
            .dump_intermediate_artifacts_to(str(art_dir))
            .build()
        )
        _ = to_edge_transform_and_lower(aten_gm, partitioner=[part])
        # Expect exactly one .tosa file in the artefact dir
        tosa_files = list(art_dir.glob("*.tosa"))
        assert (
            len(tosa_files) == 1
        ), f"Expected 1 .tosa artefact, found {len(tosa_files)} in {art_dir}"
        out_shapes = _read_tosa_outputs(tosa_files[0])
    # We use shape that is unique to output to check
    # that we preserve output order
    channel_dims = [s[-1] for s in out_shapes]
    assert channel_dims == [1, 2, 3], (
        "Outputs in .tosa do not keep author order: "
        f"expected [1, 2, 3], got {channel_dims}"
    )
