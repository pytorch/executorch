# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quantize a PyTorch model to int8 and lower it for Espressif chips with the
ESP-NN backend, producing a bundle that examples/espressif/espnn_runner can
flash and verify on the board.

Modes:
  executorch (default)  ExecuTorch program (.pte): int8 regions run in the
                        EspNNBackend delegate, anything else on the portable
                        CPU kernels. Any number of inputs and outputs.
  standalone            The whole model as one XTS3 int8 graph for the
                        espnn_runtime interpreter without the ExecuTorch
                        runtime (smallest firmware). Every operator must lower.

Examples:
  python examples/espressif/aot_espnn_compiler.py --model mobilenet_v2 \
      --calib-dir imagenette2-160/train --eval-dir imagenette2-160/val --output mv2.xtp
  python examples/espressif/aot_espnn_compiler.py --model my_models.py:make_model \
      --input-shape 1,3,96,96 --input-shape 1,10 --mode executorch --output mine.xtp
  python examples/espressif/aot_espnn_compiler.py --model mobilenet_v3_small --mode standalone \
      --input-shape 1,3,96,96 --output mv3s_96.xtb
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from executorch.backends.espressif._passes import fold_batch_norm, remove_dropout
from executorch.backends.espressif.quantizer import EspNNQuantizer
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

sys.path.insert(0, str(Path(__file__).resolve().parent))
from espnn_bundle import sqnr_db, write_bundle, xte1_vectors, xtv2_vectors  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENETTE_WNID_TO_IMAGENET = {
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


def load_model(spec: str, weights: str) -> torch.nn.Module:
    if ":" in spec and spec.split(":", 1)[0].endswith(".py"):
        path, factory = spec.split(":", 1)
        module_spec = importlib.util.spec_from_file_location("espnn_user_model", path)
        if module_spec is None or module_spec.loader is None:
            raise ValueError(f"Unable to import {path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
        return getattr(module, factory)()
    from torchvision import models

    builder = getattr(models, spec, None)
    if builder is None:
        raise ValueError(
            f"Unknown torchvision model {spec}; use path/to/file.py:factory"
        )
    return builder(weights=None if weights == "none" else weights)


class ImageSource:
    def __init__(self, shape, resize: int, mean, std, label_map):
        from torchvision import transforms

        height, width = shape[2], shape[3]
        self.transform = transforms.Compose(
            [
                transforms.Resize(resize or min(height, width)),
                transforms.CenterCrop((height, width)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.label_map = dict(IMAGENETTE_WNID_TO_IMAGENET)
        if label_map:
            self.label_map.update(json.loads(Path(label_map).read_text()))

    def load(self, directory: Path, count: int, seed: int):
        from PIL import Image

        files = sorted(
            p for p in Path(directory).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not files:
            raise ValueError(f"No images found under {directory}")
        random.Random(seed).shuffle(files)
        return [
            (
                (self.transform(Image.open(path).convert("RGB")).unsqueeze(0),),
                int(self.label_map.get(path.parent.name, -1)),
            )
            for path in files[:count]
        ]


def load_npy(path: Path, count: int, shape):
    array = np.load(path)
    if tuple(array.shape[1:]) != tuple(shape[1:]):
        raise ValueError(
            f"{path} has sample shape {array.shape[1:]}, expected {shape[1:]}"
        )
    return [
        ((torch.from_numpy(array[i : i + 1]).float(),), -1)
        for i in range(min(count, len(array)))
    ]


def make_data(args, shapes):
    image_input = len(shapes) == 1 and len(shapes[0]) == 4 and shapes[0][1] == 3
    source = None
    if image_input and (args.calib_dir or args.eval_dir):
        source = ImageSource(
            shapes[0], args.resize, args.mean, args.std, args.label_map
        )
    generator = torch.Generator().manual_seed(args.seed)

    def noise(count):
        return [
            (tuple(torch.randn(s, generator=generator) for s in shapes), -1)
            for _ in range(count)
        ]

    if args.calib_npy and len(shapes) == 1:
        calibration = load_npy(args.calib_npy, args.calib_samples, shapes[0])
    elif source is not None and args.calib_dir:
        calibration = source.load(args.calib_dir, args.calib_samples, args.seed)
    else:
        print(
            "      NOTE: no calibration data given; using Gaussian noise (accuracy will be poor)"
        )
        calibration = noise(args.calib_samples)
    if args.eval_npy and len(shapes) == 1:
        evaluation = load_npy(args.eval_npy, args.eval_samples, shapes[0])
    elif source is not None and args.eval_dir:
        evaluation = source.load(args.eval_dir, args.eval_samples, args.seed + 1)
    else:
        evaluation = calibration[: args.eval_samples]
    return calibration, evaluation


def qat_finetune(model, calibration, args):
    """Quantization-aware fine-tuning by KL distillation from the float model (no labels)."""
    import torch.nn.functional as F
    from torchao.quantization.pt2e import (
        disable_observer,
        move_exported_model_to_eval,
        move_exported_model_to_train,
    )
    from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e

    data = torch.cat([inputs[0] for inputs, _ in calibration])
    batch = min(args.qat_batch, len(data))
    student = copy.deepcopy(model)
    exported = torch.export.export(
        student,
        (data[:2],),
        dynamic_shapes=({0: torch.export.Dim("batch", min=2, max=1024)},),
        strict=True,
    ).module()
    prepared = prepare_qat_pt2e(exported, EspNNQuantizer(is_qat=True))
    for parameter in prepared.parameters():
        if parameter.is_floating_point():
            parameter.requires_grad_(True)
    parameters = [p for p in prepared.parameters() if p.requires_grad]
    total_steps = max(1, int(round(args.qat_epochs * max(1, len(data) // batch))))
    optimizer = torch.optim.SGD(parameters, lr=args.qat_lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    generator = torch.Generator().manual_seed(args.seed)
    move_exported_model_to_train(prepared)
    with torch.no_grad():
        for start in range(0, min(len(data), 16 * batch), batch):
            prepared(data[start : start + batch])
    with torch.enable_grad():
        for step in range(total_steps):
            if step == int(total_steps * 0.6):
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
            if step % max(1, total_steps // 10) == 0 or step + 1 == total_steps:
                print(f"      QAT step {step + 1}/{total_steps} loss {loss.item():.4f}")
    move_exported_model_to_eval(prepared)
    for parameter in prepared.parameters():
        parameter.requires_grad_(False)
    return prepared


def quantize(model, calibration, args):
    example = calibration[0][0]
    if args.qat_epochs > 0:
        if len(example) != 1:
            raise SystemExit("QAT is implemented for single-input models")
        print(f"[2/5] PT2E quantization-aware fine-tuning: {args.qat_epochs} epochs")
        prepared = qat_finetune(model, calibration, args)
    else:
        print(f"[2/5] PT2E quantization with {len(calibration)} calibration samples")
        prepared = prepare_pt2e(
            torch.export.export(model, example, strict=True).module(), EspNNQuantizer()
        )
        for inputs, _ in calibration:
            prepared(*inputs)
    quantized = convert_pt2e(prepared)
    return quantized, torch.export.export(quantized, example, strict=True)


def as_list(value):
    return list(value) if isinstance(value, (tuple, list)) else [value]


def evaluate(model, quantized, device_fn, evaluation):
    totals = Counter()
    worst_float = worst_pt2e = math.inf
    for inputs, label in evaluation:
        float_out = as_list(model(*inputs))
        pt2e_out = as_list(quantized(*inputs))
        device_out = as_list(device_fn(inputs))
        for f, q, d in zip(float_out, pt2e_out, device_out):
            worst_float = min(worst_float, sqnr_db(f, d.to(torch.float32)))
            worst_pt2e = min(worst_pt2e, sqnr_db(q, d.to(torch.float32)))
        if float_out[0].dim() == 2:
            float_top1 = int(float_out[0].argmax())
            device_top1 = int(device_out[0].to(torch.float32).argmax())
            totals["samples"] += 1
            totals["top1_agree_with_float"] += int(float_top1 == device_top1)
            if label >= 0:
                totals["labelled"] += 1
                totals["float_correct"] += int(float_top1 == label)
                totals["device_correct"] += int(device_top1 == label)
    print(
        f"      worst output SQNR: device vs float {worst_float:.2f} dB, "
        f"device vs PT2E int8 {worst_pt2e:.2f} dB"
    )
    if totals["samples"]:
        line = f"      top-1 agreement with float {totals['top1_agree_with_float']}/{totals['samples']}"
        if totals["labelled"]:
            line += (
                f"; accuracy float {totals['float_correct']}/{totals['labelled']}, "
                f"device {totals['device_correct']}/{totals['labelled']}"
            )
        print(line)
    return worst_float, worst_pt2e, dict(totals)


def compile_standalone(args, model, quantized, program, evaluation):
    from executorch.backends.espressif.reference import esp_nn_reference as reference
    from executorch.backends.espressif.serialization import (
        build_graph,
        parse_graph,
        serialize_graph,
    )

    print("[3/5] Lowering to one XTS3 int8 graph (standalone espnn_runtime)")
    blob = serialize_graph(build_graph(program))
    graph = parse_graph(blob)
    interpreter = reference.ReferenceInterpreter(graph)
    ops = Counter(op.opcode.name for op in graph.operations)
    arena, _ = reference.plan_activation_arena(graph)
    print(
        f"      {len(graph.operations)} operations {dict(ops)}; blob {len(blob)} bytes; "
        f"activation arena {arena} bytes"
    )

    def device_fn(inputs):
        device_inputs = [
            reference.to_device_layout(
                graph.tensors[i], reference.quantize(graph.tensors[i], x.numpy())
            )
            for i, x in zip(interpreter.inputs, inputs)
        ]
        outputs = interpreter.run_outputs(device_inputs)
        return [
            torch.from_numpy(
                reference.dequantize(
                    graph.tensors[i], reference.to_logical_layout(graph.tensors[i], out)
                )
            )
            for i, out in zip(interpreter.outputs, outputs)
        ]

    print(
        f"[4/5] Evaluating {len(evaluation)} samples: float vs PT2E int8 vs bit-exact device model"
    )
    worst_float, worst_pt2e, totals = evaluate(model, quantized, device_fn, evaluation)
    samples = [
        [x.numpy() for x in inputs] for inputs, _ in evaluation[: args.vector_samples]
    ]
    vectors = xtv2_vectors(graph, samples)
    report = {
        "mode": "standalone",
        "operations": dict(ops),
        "blob_bytes": len(blob),
        "activation_arena_bytes": arena,
        "inputs": [list(graph.tensors[i].shape) for i in graph.inputs],
        "outputs": [list(graph.tensors[i].shape) for i in graph.outputs],
    }
    return b"XTB1", blob, vectors, report, worst_float, worst_pt2e, totals


def compile_executorch(args, model, quantized, program, evaluation):
    from executorch.backends.espressif import to_espnn_edge
    from executorch.backends.espressif.reference import ProgramEmulator
    from executorch.exir.lowered_backend_module import LoweredBackendModule

    print("[3/5] Lowering through ExecuTorch: EspNN delegates + portable CPU fallback")
    edge, partition_report = to_espnn_edge(
        program, allow_cpu_fallback=not args.no_cpu_fallback
    )
    exported = edge.exported_program()
    delegates = [
        m
        for _, m in exported.graph_module.named_children()
        if isinstance(m, LoweredBackendModule)
    ]
    pte = edge.to_executorch().buffer
    cpu_ops = partition_report["cpu_ops"]
    print(
        f"      {len(delegates)} delegate(s), {partition_report['delegated_ops']} delegated operators, "
        f"{len(cpu_ops)} operators on the CPU; .pte {len(pte)} bytes"
    )
    for target, count in Counter(op["target"] for op in cpu_ops).most_common(12):
        reason = next(op["reason"] for op in cpu_ops if op["target"] == target)
        print(f"      CPU: {target} x{count} ({reason})")
    emulator = ProgramEmulator(exported)
    print(
        f"[4/5] Evaluating {len(evaluation)} samples: float vs PT2E int8 vs emulated device program"
    )
    worst_float, worst_pt2e, totals = evaluate(
        model, quantized, lambda inputs: as_list(emulator(*inputs)), evaluation
    )
    rows = [
        (list(inputs), as_list(emulator(*inputs)))
        for inputs, _ in evaluation[: args.vector_samples]
    ]
    exact = not cpu_ops
    vectors = xte1_vectors(
        rows, exact, args.device_min_sqnr, crc=exact and args.crc_outputs
    )
    if args.pte_output:
        args.pte_output.write_bytes(pte)
    report = {
        "mode": "executorch",
        "pte_bytes": len(pte),
        "delegates": len(delegates),
        "partitions": partition_report["partitions"],
        "delegated_ops": partition_report["delegated_ops"],
        "cpu_ops": cpu_ops,
        "exact_required": exact,
        "min_sqnr_db": args.device_min_sqnr,
        "inputs": [list(t.shape) for t in rows[0][0]],
        "outputs": [list(t.shape) for t in rows[0][1]],
    }
    return b"XTP1", pte, vectors, report, worst_float, worst_pt2e, totals


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        required=True,
        help="torchvision model name or path/to/file.py:factory",
    )
    parser.add_argument(
        "--weights", default="DEFAULT", help="torchvision weights name, or 'none'"
    )
    parser.add_argument(
        "--input-shape",
        action="append",
        help="input shape such as 1,3,224,224; repeat for several inputs",
    )
    parser.add_argument(
        "--mode", choices=("executorch", "standalone"), default="executorch"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="bundle to write (.xtp or .xtb)"
    )
    parser.add_argument(
        "--name", help="entry name inside the bundle (default: model name)"
    )
    parser.add_argument(
        "--pte-output", type=Path, help="executorch mode: also write the plain .pte"
    )
    parser.add_argument("--calib-dir", type=Path)
    parser.add_argument("--calib-npy", type=Path)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--eval-dir", type=Path)
    parser.add_argument("--eval-npy", type=Path)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument(
        "--vector-samples",
        type=int,
        default=4,
        help="samples embedded in the bundle for on-board checking",
    )
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--mean", type=float, nargs=3, default=[0.485, 0.456, 0.406])
    parser.add_argument("--std", type=float, nargs=3, default=[0.229, 0.224, 0.225])
    parser.add_argument(
        "--label-map", help="JSON mapping image folder names to class indices"
    )
    parser.add_argument("--qat-epochs", type=float, default=0.0)
    parser.add_argument("--qat-lr", type=float, default=2e-4)
    parser.add_argument("--qat-batch", type=int, default=16)
    parser.add_argument(
        "--no-cpu-fallback",
        action="store_true",
        help="executorch mode: fail if any operator would run on the CPU",
    )
    parser.add_argument(
        "--device-min-sqnr",
        type=float,
        default=30.0,
        help="on-board pass threshold (dB vs host) for programs with CPU operators",
    )
    parser.add_argument(
        "--crc-outputs",
        action="store_true",
        help="store CRC32 instead of full expected outputs (large outputs)",
    )
    parser.add_argument(
        "--min-sqnr",
        type=float,
        default=-math.inf,
        help="fail when the device model's SQNR against float is lower",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    started = time.time()
    shapes = [
        tuple(int(v) for v in s.split(","))
        for s in (args.input_shape or ["1,3,224,224"])
    ]
    print(f"[1/5] Loading {args.model}")
    model = remove_dropout(
        fold_batch_norm(load_model(args.model, args.weights).eval())[0]
    ).eval()
    calibration, evaluation = make_data(args, shapes)
    quantized, program = quantize(model, calibration, args)
    compile_fn = compile_standalone if args.mode == "standalone" else compile_executorch
    magic, blob, vectors, report, worst_float, worst_pt2e, totals = compile_fn(
        args, model, quantized, program, evaluation
    )
    print("[5/5] Writing artifacts")
    name = args.name or Path(args.model.split(":")[-1]).stem
    size = write_bundle(args.output, magic, [(name, blob, vectors)])
    report.update(
        {
            "name": name,
            "model": args.model,
            "bundle": str(args.output),
            "bundle_bytes": size,
            "calibration_samples": len(calibration),
            "qat_epochs": args.qat_epochs,
            "sqnr_vs_float_db": worst_float,
            "sqnr_vs_pt2e_db": worst_pt2e,
            "evaluation": totals,
        }
    )
    args.output.with_suffix(".json").write_text(json.dumps([report], indent=2))
    print(
        f"      bundle {args.output} ({size} bytes); report {args.output.with_suffix('.json')}"
    )
    print(
        f"      embed it: CONFIG_ESPNN_RUNNER_BUNDLE (espnn_runner) or CONFIG_CAMERA_STREAM_MODEL (camera_stream) = {args.output.resolve()}"
    )
    print(f"      elapsed {time.time() - started:.1f} s")
    if worst_float < args.min_sqnr:
        print(
            f"ESPNN COMPILE: FAIL (SQNR {worst_float:.2f} dB below {args.min_sqnr:.2f} dB)"
        )
        return 1
    print("ESPNN COMPILE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
