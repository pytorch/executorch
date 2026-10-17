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


class HeadsOnly(nn.Module):
    """Executes only the heads of YOLO_NAS_S (takes neck output as input)."""

    def __init__(self, model):
        super().__init__()
        self.heads = model.heads

    def forward(self, x):
        return self.heads(x)


# Granular head component wrappers for debugging


class SingleHead(nn.Module):
    """Executes a single detection head (head1, head2, or head3)."""

    def __init__(self, model, head_idx):
        super().__init__()
        self.head = getattr(model.heads, f"head{head_idx + 1}")

    def forward(self, x):
        return self.head(x)


class HeadStemOnly(nn.Module):
    """Executes only the stem of a single head."""

    def __init__(self, model, head_idx):
        super().__init__()
        head = getattr(model.heads, f"head{head_idx + 1}")
        self.stem = head.stem

    def forward(self, x):
        return self.stem(x)


class HeadStemAndClsConvs(nn.Module):
    """Executes stem + cls_convs of a single head."""

    def __init__(self, model, head_idx):
        super().__init__()
        head = getattr(model.heads, f"head{head_idx + 1}")
        self.stem = head.stem
        self.cls_convs = head.cls_convs

    def forward(self, x):
        x = self.stem(x)
        return self.cls_convs(x)


class HeadStemAndRegConvs(nn.Module):
    """Executes stem + reg_convs of a single head."""

    def __init__(self, model, head_idx):
        super().__init__()
        head = getattr(model.heads, f"head{head_idx + 1}")
        self.stem = head.stem
        self.reg_convs = head.reg_convs

    def forward(self, x):
        x = self.stem(x)
        return self.reg_convs(x)


class HeadClsPred(nn.Module):
    """Executes stem + cls_convs + cls_pred of a single head."""

    def __init__(self, model, head_idx):
        super().__init__()
        head = getattr(model.heads, f"head{head_idx + 1}")
        self.stem = head.stem
        self.cls_convs = head.cls_convs
        self.cls_pred = head.cls_pred

    def forward(self, x):
        x = self.stem(x)
        x = self.cls_convs(x)
        return self.cls_pred(x)


class HeadRegPred(nn.Module):
    """Executes stem + reg_convs + reg_pred of a single head."""

    def __init__(self, model, head_idx):
        super().__init__()
        head = getattr(model.heads, f"head{head_idx + 1}")
        self.stem = head.stem
        self.reg_convs = head.reg_convs
        self.reg_pred = head.reg_pred

    def forward(self, x):
        x = self.stem(x)
        x = self.reg_convs(x)
        return self.reg_pred(x)


class AllHeadsRawOutputs(nn.Module):
    """Executes all heads but returns raw outputs before DFL decoding."""

    def __init__(self, model):
        super().__init__()
        self.head1 = model.heads.head1
        self.head2 = model.heads.head2
        self.head3 = model.heads.head3

    def forward(self, feats):
        reg1, cls1 = self.head1(feats[0])
        reg2, cls2 = self.head2(feats[1])
        reg3, cls3 = self.head3(feats[2])
        return reg1, cls1, reg2, cls2, reg3, cls3


class HeadsDFLDecode(nn.Module):
    """Executes DFL decoding: softmax + proj_conv reduction on reg outputs."""

    def __init__(self, model):
        super().__init__()
        self.reg_max = model.heads.reg_max
        self.register_buffer("proj_conv", model.heads.proj_conv.clone())

    def forward(self, reg_distri):
        # reg_distri shape: [B, 4*(reg_max+1), H, W]
        b, _, h, w = reg_distri.shape
        height_mul_width = h * w

        # Reshape for DFL
        reg_dist_reduced = reg_distri.reshape(b, 4, self.reg_max + 1, height_mul_width)
        reg_dist_reduced = reg_dist_reduced.permute(0, 2, 3, 1)  # [B, reg_max+1, H*W, 4]

        # Softmax and reduce
        reg_dist_reduced = torch.nn.functional.softmax(reg_dist_reduced, dim=1)
        reg_dist_reduced = reg_dist_reduced * self.proj_conv
        reg_dist_reduced = reg_dist_reduced.sum(dim=1)  # [B, H*W, 4]

        return reg_dist_reduced


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

    # Get intermediate outputs for testing components in isolation
    with torch.no_grad():
        backbone_out = full_model.backbone(example_inputs[0])
        neck_out = full_model.neck(backbone_out)

        # Get individual feature maps for each head (stride 8, 16, 32)
        feat1 = neck_out[0]  # For head1 (stride 8)
        feat2 = neck_out[1]  # For head2 (stride 16)
        feat3 = neck_out[2]  # For head3 (stride 32)

        # Get raw head outputs for DFL decode testing
        reg1, _ = full_model.heads.head1(feat1)

    heads_inputs = (neck_out,)

    # High-level partial models
    partial_models = [
        ("Backbone Only", BackboneOnly(full_model), example_inputs),
        ("Backbone + Neck", BackboneAndNeck(full_model), example_inputs),
    ]

    # Granular head tests - test head1 components (stride 8, largest feature map)
    head_idx = 0  # Test head1
    head_models = [
        (f"Head1 Stem Only", HeadStemOnly(full_model, head_idx), (feat1,)),
        (f"Head1 Stem+ClsConvs", HeadStemAndClsConvs(full_model, head_idx), (feat1,)),
        (f"Head1 Stem+RegConvs", HeadStemAndRegConvs(full_model, head_idx), (feat1,)),
        (f"Head1 ClsPred", HeadClsPred(full_model, head_idx), (feat1,)),
        (f"Head1 RegPred", HeadRegPred(full_model, head_idx), (feat1,)),
        (f"Head1 Full", SingleHead(full_model, head_idx), (feat1,)),
    ]

    # Test individual heads
    individual_heads = [
        ("Head1 (stride 8)", SingleHead(full_model, 0), (feat1,)),
        ("Head2 (stride 16)", SingleHead(full_model, 1), (feat2,)),
        ("Head3 (stride 32)", SingleHead(full_model, 2), (feat3,)),
    ]

    # Post-head processing
    post_head_models = [
        ("All Heads Raw", AllHeadsRawOutputs(full_model), heads_inputs),
        ("DFL Decode", HeadsDFLDecode(full_model), (reg1,)),
        ("Heads Only (full)", HeadsOnly(full_model), heads_inputs),
    ]

    # Full model
    full_model_tests = [
        ("Full Model", FullModel(full_model), example_inputs),
    ]

    all_tests = partial_models + head_models + individual_heads + post_head_models + full_model_tests

    results = {}
    first_failure = None

    for name, model, inputs in all_tests:
        passed = test_partial_model(name, model, inputs)
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
