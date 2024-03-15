# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
from typing import Tuple

import torch

from d2go.projects.ego_ocr.recognition.recognition_runner import RecognitionRunner
from d2go.setup import setup_after_launch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import EdgeProgramManager, to_edge
from torch.export import export, ExportedProgram

ctypes.CDLL("libvulkan.so.1")
torch.ops.load_library("//executorch/backends/vulkan:vulkan_backend_lib")
torch.ops.load_library("//executorch/kernels/portable:custom_ops_generated_lib")

from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)
from executorch.extension.pytree import tree_flatten


def lower_module_and_inference(
    model: torch.nn.Module,
    sample_inputs: Tuple[torch.Tensor],
):
    program: ExportedProgram = export(model, sample_inputs)
    edge_program: EdgeProgramManager = to_edge(program)
    edge_program = edge_program.to_backend(VulkanPartitioner())

    executorch_program = edge_program.to_executorch()
    executorch_module = _load_for_executorch_from_buffer(executorch_program.buffer)
    # pyre-fixme[16]: Module `pytree` has no attribute `tree_flatten`.
    inputs_flattened, _ = tree_flatten(sample_inputs)

    model_output = executorch_module.run_method("forward", tuple(inputs_flattened))
    if model_output is not None:
        print("Success output from ET model")


def export_ocr_recognizer() -> None:
    cfg = RecognitionRunner.get_default_cfg()
    cfg.merge_from_file(
        "manifold://fai4ar_supar/tree/ocr/workflow/ashishvs/20231011/f489950547/e2e_train/trained_model_configs/model_final.yaml"
    )
    cfg.MODEL.DEVICE = "cpu"
    cfg.D2GO_DATA.TEST.MAX_IMAGES = 1
    cfg.MODEL.WEIGHTS = "manifold://fai4ar_supar/tree/ocr/workflow/ashishvs/20231011/f489950547/e2e_train/last.ckpt"
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.QUANTIZATION.BACKEND = "qnnpack"
    cfg.QUANTIZATION.PTQ.CALIBRATION_NUM_IMAGES = 32
    cfg.QUANTIZATION.EAGER_MODE = True
    cfg.DATASETS.TRAIN = ["egotext_gbd_without_unk_rcg"]
    cfg.DATASETS.TEST = ["egotext_gbd_without_unk_rcg"]
    cfg.MODEL_EMA.USE_EMA_WEIGHTS_FOR_EVAL_ONLY = True
    cfg.EGO_OCR.RECOGNITION.IN_CHANNELS = 3
    cfg.EGO_OCR.RECOGNITION.INPUT_FORMAT = "RGB"

    output_dir = "/tmp/ocr_recog_temp"

    runner = setup_after_launch(cfg, output_dir, RecognitionRunner)
    assert runner is not None
    recog_model_full = runner.build_model(cfg, eval_only=True)
    recog_model_exp_config = recog_model_full.prepare_for_export(
        cfg, inputs=None, predictor_type=""
    )
    recog_model = recog_model_exp_config.model

    inputs: Tuple[torch.Tensor] = (
        torch.rand(
            3,
            cfg.EGO_OCR.RECOGNITION.IMAGE_HEIGHT,
            cfg.EGO_OCR.RECOGNITION.MAX_WIDTH,
        ).unsqueeze(0),
    )

    lower_module_and_inference(recog_model, inputs)


def test_vulkan_add() -> None:
    class AddModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

    add_module = AddModule()
    model_inputs = (
        torch.rand(size=(2, 3), dtype=torch.float32),
        torch.rand(size=(2, 3), dtype=torch.float32),
    )
    lower_module_and_inference(add_module, model_inputs)


def main() -> None:
    """Script to export OCR recognizer to Executorch Vulkan backend"""

    export_ocr_recognizer()


if __name__ == "__main__":
    main()
