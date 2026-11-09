# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quantize a torchvision ImageNet classifier for the camera demo and lower it
to an ExecuTorch program (.pte) with ESP-NN delegates.

Calibrates on an image folder, optionally fine-tunes with quantization-aware
training (distillation from the float model, no labels needed), and reports
top-1 accuracy of the lowered program as the board computes it. Also writes
<output>.ops.txt, the CPU operators to select when building the runtime.

    python export_model.py --calib-dir imagenette2-160/train \\
        --eval-dir imagenette2-160/val --qat-epochs 5 --qat-lr 1e-3 \\
        --output models/mobilenet_v3_small_224.pte

MobileNetV3-Small needs QAT; plain PTQ loses most of its accuracy.
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from executorch.backends.espressif import EspNNQuantizer, to_espnn_edge
from executorch.backends.espressif.reference import ProgramEmulator
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from torchao.quantization.pt2e import (
    disable_observer,
    move_exported_model_to_eval,
    move_exported_model_to_train,
)
from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Imagenette folder names and their ImageNet-1k class indices.
IMAGENETTE_TO_IMAGENET = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_images(directory: Path, count: int, size: int, seed: int):
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize(size * 256 // 224),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    files = sorted(
        p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise SystemExit(f"no images found under {directory}")
    random.Random(seed).shuffle(files)
    images, labels = [], []
    for path in files[:count]:
        images.append(transform(Image.open(path).convert("RGB")))
        labels.append(IMAGENETTE_TO_IMAGENET.get(path.parent.name, -1))
    return torch.stack(images), labels


def qat_prepare(model, data, args):
    batch = min(args.qat_batch, len(data))
    exported = torch.export.export(
        copy.deepcopy(model),
        (data[:2],),
        dynamic_shapes=({0: torch.export.Dim("batch", min=2, max=1024)},),
        strict=True,
    ).module()
    prepared = prepare_qat_pt2e(exported, EspNNQuantizer(is_qat=True))
    parameters = [p for p in prepared.parameters() if p.is_floating_point()]
    for parameter in parameters:
        parameter.requires_grad_(True)
    steps = max(1, round(args.qat_epochs * max(1, len(data) // batch)))
    optimizer = torch.optim.SGD(parameters, lr=args.qat_lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    generator = torch.Generator().manual_seed(args.seed)
    move_exported_model_to_train(prepared)
    with torch.no_grad():
        for start in range(0, min(len(data), 16 * batch), batch):
            prepared(data[start : start + batch])
    with torch.enable_grad():
        for step in range(steps):
            if step == int(steps * 0.6):
                prepared.apply(disable_observer)
            inputs = data[torch.randint(0, len(data), (batch,), generator=generator)]
            if bool(torch.rand(1, generator=generator) < 0.5):
                inputs = inputs.flip(3)
            with torch.no_grad():
                teacher = F.softmax(model(inputs), dim=1)
            loss = F.kl_div(
                F.log_softmax(prepared(inputs), dim=1), teacher, reduction="batchmean"
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % max(1, steps // 10) == 0 or step + 1 == steps:
                print(f"QAT step {step + 1}/{steps} loss {loss.item():.4f}")
    move_exported_model_to_eval(prepared)
    for parameter in parameters:
        parameter.requires_grad_(False)
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mobilenet_v3_small")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--calib-samples", type=int, default=1024)
    parser.add_argument("--eval-dir", type=Path)
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--qat-epochs", type=float, default=0.0)
    parser.add_argument("--qat-lr", type=float, default=1e-3)
    parser.add_argument("--qat-batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("models/mobilenet_v3_small_224.pte")
    )
    args = parser.parse_args()

    from torchvision import models

    model = getattr(models, args.model)(weights="DEFAULT").eval()
    calibration, _ = load_images(
        args.calib_dir, args.calib_samples, args.size, args.seed
    )
    example = (calibration[:1],)

    if args.qat_epochs > 0:
        prepared = qat_prepare(model, calibration, args)
    else:
        prepared = prepare_pt2e(
            torch.export.export(model, example, strict=True).module(),
            EspNNQuantizer(),
        )
        with torch.no_grad():
            for image in calibration:
                prepared(image.unsqueeze(0))
    quantized = convert_pt2e(prepared)

    edge, report = to_espnn_edge(torch.export.export(quantized, example, strict=True))
    print(
        f"{report['partitions']} ESP-NN partition(s), "
        f"{report['delegated_ops']} delegated operators, "
        f"{len(report['cpu_ops'])} on the CPU"
    )

    if args.eval_dir is not None:
        images, labels = load_images(
            args.eval_dir, args.eval_samples, args.size, args.seed + 1
        )
        emulator = ProgramEmulator(edge.exported_program())
        float_correct = device_correct = agree = 0
        with torch.no_grad():
            for image, label in zip(images, labels):
                image = image.unsqueeze(0)
                float_top1 = int(model(image).argmax())
                device_top1 = int(emulator(image).to(torch.float32).argmax())
                float_correct += int(float_top1 == label)
                device_correct += int(device_top1 == label)
                agree += int(float_top1 == device_top1)
        print(
            f"top-1 on {len(labels)} images: float {float_correct}, "
            f"int8 program {device_correct}, agreement {agree}"
        )

    program = edge.to_executorch(
        ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
        )
    )
    plan = program.executorch_program.execution_plan[0]
    operators = sorted(
        f"{op.name}.{op.overload}"
        for op in plan.operators
        if not op.name.startswith("quantized_decomposed::")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(program.buffer)
    ops_path = args.output.with_suffix(".ops.txt")
    ops_path.write_text(",".join(operators) + "\n")
    print(f"wrote {args.output} ({len(program.buffer)} bytes) and {ops_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
