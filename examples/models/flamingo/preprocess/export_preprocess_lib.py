# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.exir.program._program import ExecutorchProgramManager

from executorch.extension.llm.custom_ops import preprocess_custom_ops  # noqa

from torch.export import Dim, ExportedProgram
from torchtune.models.clip.inference._transform import _CLIPImageTransform


def get_example_inputs() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = torch.ones(3, 800, 600)
    target_size = torch.tensor([448, 336])
    canvas_size = torch.tensor([448, 448])
    return (image, target_size, canvas_size)


def get_dynamic_shapes() -> Dict[str, Dict[int, Dim]]:
    img_h = Dim("img_h", min=1, max=4000)
    img_w = Dim("img_w", min=1, max=4000)

    dynamic_shapes = {
        "image": {1: img_h, 2: img_w},
        "target_size": None,
        "canvas_size": None,
    }
    return dynamic_shapes


def export_preprocess(
    resample: str = "bilinear",
    image_mean: Optional[List[float]] = None,
    image_std: Optional[List[float]] = None,
    max_num_tiles: int = 4,
    tile_size: int = 224,
    antialias: bool = False,
) -> ExportedProgram:

    # Instantiate eager model.
    image_transform_model = _CLIPImageTransform(
        resample=resample,
        image_mean=image_mean,
        image_std=image_std,
        max_num_tiles=max_num_tiles,
        tile_size=tile_size,
        antialias=antialias,
    )

    # Replace non-exportable ops with custom ops.
    image_transform_model.tile_crop = torch.ops.preprocess.tile_crop.default

    # Export.
    example_inputs = get_example_inputs()
    dynamic_shapes = get_dynamic_shapes()
    ep = torch.export.export(
        image_transform_model,
        example_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    return ep


def lower_to_executorch_preprocess(
    exported_program: ExportedProgram,
) -> ExecutorchProgramManager:
    edge_program = to_edge(
        exported_program, compile_config=EdgeCompileConfig(_check_ir_validity=False)
    )

    et_program = edge_program.to_executorch(
        ExecutorchBackendConfig(
            sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
        )
    )
    return et_program
