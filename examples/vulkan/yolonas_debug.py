# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Standalone export script for YOLO_NAS_S model with Vulkan delegate
# Includes partial model wrappers for debugging incorrect outputs

import logging
import urllib

import executorch.backends.vulkan.test.utils as test_utils
import torch
import torch.nn as nn
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program
from torch.export import export

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


# Partial model wrappers for progressive debugging


class BackboneOnly(nn.Module):
    """Executes only the backbone of YOLO_NAS_S."""

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone

    def forward(self, x):
        return self.backbone(x)


class BackboneAndNeck(nn.Module):
    """Executes backbone + neck of YOLO_NAS_S."""

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return x


class FullModel(nn.Module):
    """Executes the full YOLO_NAS_S model (backbone + neck + heads)."""

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck
        self.heads = model.heads

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return self.heads(x)


def get_model():
    try:
        from super_gradients.common.object_names import Models
        from super_gradients.training import models
    except ImportError:
        raise ImportError(
            "Please install super-gradients: pip install super-gradients"
        )

    return models.get(Models.YOLO_NAS_S, pretrained_weights="coco")


def get_dog_image_tensor(image_size=640):
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg",
    )
    try:
        urllib.URLopener().retrieve(url, filename)
    except Exception:
        urllib.request.urlretrieve(url, filename)

    from PIL import Image
    from torchvision import transforms

    input_image = Image.open(filename).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return (input_batch,)


def test_partial_model(name, model, example_inputs, atol=1e-4, rtol=1e-4):
    """Export and test a partial model, returning True if outputs match."""
    logging.info(f"\n{'='*60}")
    logging.info(f"Testing: {name}")
    logging.info(f"{'='*60}")

    model.eval()

    logging.info(f"Exporting {name} with torch.export...")
    program = export(model, example_inputs, strict=True)

    logging.info(f"Lowering {name} to Vulkan delegate...")
    edge_program = to_edge_transform_and_lower(
        program,
        partitioner=[VulkanPartitioner({})],
    )

    exec_prog = edge_program.to_executorch()

    output_filename = f"yolonas_{name.lower().replace(' ', '_')}_vulkan"
    save_pte_program(exec_prog, output_filename, ".")
    logging.info(f"Saved as {output_filename}.pte")

    logging.info(f"Running correctness check for {name}...")
    test_result = test_utils.run_and_check_output(
        reference_model=model,
        executorch_program=exec_prog,
        sample_inputs=example_inputs,
        atol=atol,
        rtol=rtol,
    )

    if test_result:
        logging.info(f"[PASS] {name} - outputs match reference")
    else:
        logging.error(f"[FAIL] {name} - outputs do NOT match reference")

    return test_result


def main():
    logging.info("Loading YOLO_NAS_S model...")
    full_model = get_model()
    full_model.eval()

    logging.info("Preparing sample input (640x640 dog image)...")
    example_inputs = get_dog_image_tensor(640)

    # Create partial models for progressive testing
    partial_models = [
        ("Backbone Only", BackboneOnly(full_model)),
        ("Backbone + Neck", BackboneAndNeck(full_model)),
        ("Full Model", FullModel(full_model)),
    ]

    results = {}
    first_failure = None

    for name, model in partial_models:
        passed = test_partial_model(name, model, example_inputs)
        results[name] = passed
        if not passed and first_failure is None:
            first_failure = name

    # Summary
    logging.info(f"\n{'='*60}")
    logging.info("SUMMARY")
    logging.info(f"{'='*60}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logging.info(f"  {name}: {status}")

    if first_failure:
        logging.info(f"\nFirst failure detected at: {first_failure}")
        logging.info("Investigate this component for incorrect Vulkan outputs.")
    else:
        logging.info("\nAll partial models passed!")


if __name__ == "__main__":
    with torch.no_grad():
        main()
