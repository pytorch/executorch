#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Compare YOLO26 detections between:
1) Exported CUDA hybrid ExecuTorch model (.pte)
2) Downloaded XNNPACK ExecuTorch model (.pte from Hugging Face)

This script uses ExecuTorch Python runtime (pybind API) for model execution.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch
from executorch.runtime import Runtime
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from ultralytics.utils import ops


@dataclass
class Detection:
    cls: int
    name: str
    conf: float
    xyxy: np.ndarray  # [x1, y1, x2, y2] in original image space

    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy
        return np.array(
            [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1],
            dtype=float,
        )


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def ensure_yolo_predictor(model_name: str, img_bgr: np.ndarray, rect: bool = False) -> YOLO:
    model = YOLO(model_name)
    # Initialize predictor internals (imgsz, stride, transforms, args, ...)
    model.predict(img_bgr, imgsz=(640, 640), rect=rect, device="cpu", verbose=False)
    return model


def preprocess_with_yolo(model: YOLO, img_bgr: np.ndarray) -> torch.Tensor:
    x = model.predictor.preprocess([img_bgr]).contiguous()
    return x


def load_program_method(runtime: Runtime, pte_path: Path):
    program = runtime.load_program(str(pte_path))
    method = program.load_method("forward")
    if method is None:
        raise RuntimeError(f"Failed to load forward method from {pte_path}")
    return method


def load_forward_module(runtime: Runtime, pte_path: Path, ptd_path: Path):
    # Runtime.load_program only consumes .pte. Use low-level pybind API for .ptd.
    return runtime._legacy_module._load_for_executorch(
        str(pte_path),
        str(ptd_path),
    )


def tensor_from_output(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).cpu()
    return torch.as_tensor(value).cpu()


def parse_packed_detections(
    output: torch.Tensor,
    conf_threshold: float,
    names: dict,
    input_hw: Sequence[int],
    orig_hw: Sequence[int],
    packed_format: str = "xyxy",
) -> List[Detection]:
    # Expect [B, N, >=6], columns are [box4, conf, cls, ...]
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[2] < 6:
        raise ValueError(f"Not packed detections: got shape {tuple(output.shape)}")

    det = output[0, :, :6].clone().float()
    boxes = det[:, :4]
    conf = det[:, 4]
    cls = det[:, 5].round().to(torch.int64)

    if packed_format == "xywh":
        xywh = boxes
        xyxy = torch.empty_like(xywh)
        xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
        xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
        xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
        xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0
        boxes = xyxy
    elif packed_format != "xyxy":
        raise ValueError(f"Unsupported packed_format: {packed_format}")

    boxes = ops.scale_boxes(input_hw, boxes, orig_hw)

    keep = conf >= conf_threshold
    boxes = boxes[keep].cpu().numpy()
    conf = conf[keep].cpu().numpy()
    cls = cls[keep].cpu().numpy()

    detections = []
    for b, c, k in zip(boxes, conf, cls):
        detections.append(
            Detection(
                cls=int(k),
                name=names.get(int(k), str(int(k))),
                conf=float(c),
                xyxy=b.astype(float),
            )
        )

    detections.sort(key=lambda d: d.conf, reverse=True)
    return detections


def parse_with_ultralytics_postprocess(
    predictor,
    output: torch.Tensor,
    input_tensor: torch.Tensor,
    orig_img_bgr: np.ndarray,
) -> List[Detection]:
    # For raw head outputs, reuse YOLO predictor postprocess (NMS/scaling).
    results = predictor.postprocess(output, input_tensor, [orig_img_bgr])
    if not results:
        return []
    boxes = results[0].boxes
    detections = []
    for b in boxes:
        cls = int(b.cls[0].item())
        detections.append(
            Detection(
                cls=cls,
                name=results[0].names.get(cls, str(cls)),
                conf=float(b.conf[0].item()),
                xyxy=b.xyxy[0].cpu().numpy().astype(float),
            )
        )
    detections.sort(key=lambda d: d.conf, reverse=True)
    return detections


def run_executorch_model(
    runtime: Runtime,
    pte_path: Path,
    input_tensor: torch.Tensor,
    predictor,
    names: dict,
    conf_threshold: float,
    packed_format: str,
    input_device: str = "cpu",
    ptd_path: Optional[Path] = None,
) -> List[Detection]:
    method = None
    module = None
    if ptd_path is not None:
        if not ptd_path.exists():
            raise FileNotFoundError(f"External data file not found: {ptd_path}")
        module = load_forward_module(runtime, pte_path, ptd_path)
    else:
        method = load_program_method(runtime, pte_path)
    if input_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA input requested but torch.cuda.is_available() is False"
            )
        safe_input = input_tensor.to("cuda").contiguous()
    elif input_device == "cpu":
        safe_input = input_tensor.to("cpu").contiguous()
    else:
        raise ValueError(f"Unsupported input_device: {input_device}")

    try:
        if module is not None:
            outputs = module.forward((safe_input,))
        else:
            outputs = method.execute((safe_input,))
    except Exception as e:
        raise RuntimeError(
            f"method.execute() failed for {pte_path} with input shape={tuple(safe_input.shape)} "
            f"device={safe_input.device} contiguous={safe_input.is_contiguous()}: {e}"
        ) from e
    if outputs is None or len(outputs) == 0:
        raise RuntimeError(f"No outputs from {pte_path}")
    output0 = tensor_from_output(outputs[0])

    input_hw = input_tensor.shape[2:]
    orig_hw = predictor.batch[1][0].shape[:2] if predictor.batch is not None else None
    if orig_hw is None:
        raise RuntimeError("Predictor batch context not initialized")

    # Prefer packed parsing only for compact last-dim outputs such as [1, N, 6].
    is_compact_packed = (
        output0.ndim == 3
        and output0.shape[0] == 1
        and 6 <= output0.shape[2] <= 16
        and output0.shape[1] >= 1
    )
    if is_compact_packed:
        try:
            return parse_packed_detections(
                output0,
                conf_threshold=conf_threshold,
                names=names,
                input_hw=input_hw,
                orig_hw=orig_hw,
                packed_format=packed_format,
            )
        except Exception:
            # Fall back to Ultralytics postprocess for raw/logit-style tensors.
            pass

    return parse_with_ultralytics_postprocess(
        predictor=predictor,
        output=output0,
        input_tensor=input_tensor,
        orig_img_bgr=predictor.batch[1][0],
    )


def download_xnnpack_model(repo_id: str, filename: str, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
    )
    return Path(downloaded)


def compare_detections(
    left_name: str,
    left: List[Detection],
    right_name: str,
    right: List[Detection],
    topk: int,
) -> None:
    print(f"\n{left_name}: {len(left)} detections")
    print(f"{right_name}: {len(right)} detections")

    if len(left) == 0 or len(right) == 0:
        print("Cannot compare: one side has no detections.")
        return

    print(f"\nTop-{min(topk, len(left))} {left_name} detections vs best class-matched {right_name}:")
    ious = []
    conf_diffs = []
    for i, d in enumerate(left[:topk], start=1):
        best: Optional[Detection] = None
        best_iou = -1.0
        for r in right:
            if r.cls != d.cls:
                continue
            v = iou_xyxy(d.xyxy, r.xyxy)
            if v > best_iou:
                best_iou = v
                best = r

        dx, dy, dw, dh = d.xywh()
        if best is None:
            print(
                f"{i:02d}. {d.name} conf={d.conf:.4f} "
                f"xywh=[{dx:.1f},{dy:.1f},{dw:.1f},{dh:.1f}] | no class match"
            )
            continue

        bx, by, bw, bh = best.xywh()
        ious.append(best_iou)
        conf_diffs.append(abs(d.conf - best.conf))
        print(
            f"{i:02d}. {d.name} conf={d.conf:.4f} "
            f"xywh=[{dx:.1f},{dy:.1f},{dw:.1f},{dh:.1f}] "
            f"| {right_name} conf={best.conf:.4f} "
            f"xywh=[{bx:.1f},{by:.1f},{bw:.1f},{bh:.1f}] "
            f"| IoU={best_iou:.4f}"
        )

    if ious:
        print(
            f"\nIoU stats: mean={np.mean(ious):.4f}, "
            f"min={np.min(ious):.4f}, max={np.max(ious):.4f}"
        )
        print(f"Mean |conf diff|: {np.mean(conf_diffs):.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare YOLO26 CUDA vs XNNPACK ExecuTorch outputs")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("/home/larryliu/miniconda3/envs/executorch/lib/python3.12/site-packages/ultralytics/assets/bus.jpg"),
        help="Input image path (default: Ultralytics bus.jpg)",
    )
    parser.add_argument(
        "--ultralytics-model",
        type=str,
        default="yolo26n.pt",
        help="Ultralytics model for preprocessing/postprocessing context",
    )
    parser.add_argument(
        "--cuda-pte",
        type=Path,
        default=Path("/tmp/yolo_hybrid_out/yolo26n/model.pte"),
        help="Path to exported CUDA hybrid model.pte",
    )
    parser.add_argument(
        "--cuda-ptd",
        type=Path,
        default=Path("/tmp/yolo_hybrid_out/yolo26n/aoti_cuda_blob.ptd"),
        help="Path to exported CUDA external data (.ptd)",
    )
    parser.add_argument(
        "--xnnpack-pte",
        type=Path,
        default=None,
        help="Local path to XNNPACK pte; if not set, downloads from Hugging Face",
    )
    parser.add_argument(
        "--xnnpack-repo",
        type=str,
        default="larryliu0820/yolo26n-ExecuTorch-XNNPACK",
        help="Hugging Face repo for XNNPACK model",
    )
    parser.add_argument(
        "--xnnpack-file",
        type=str,
        default="yolo26n_dynamic_xnnpack.pte",
        help="Filename in Hugging Face repo",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=Path("/tmp/hf_yolo26n_xnnpack"),
        help="Directory for downloaded Hugging Face artifacts",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="How many top detections to compare",
    )
    parser.add_argument(
        "--packed-format",
        type=str,
        choices=["xyxy", "xywh"],
        default="xyxy",
        help="Interpretation for packed [1,N,6] outputs",
    )
    parser.add_argument(
        "--rect",
        action="store_true",
        help=(
            "Use Ultralytics rect preprocessing (can produce non-square inputs like 640x480). "
            "Default is off (square 640x640) to match current CUDA export constraints."
        ),
    )
    parser.add_argument(
        "--allow-nonsquare-cuda-input",
        action="store_true",
        help=(
            "Allow non-square input for CUDA model. "
            "Current CUDA export in yolo26_cuda_hybrid_export.py constrains H==W."
        ),
    )
    parser.add_argument(
        "--cuda-input-device",
        type=str,
        choices=["cuda", "cpu"],
        default="cpu",
        help=(
            "Tensor device used for CUDA model execute(). "
            "Default 'cpu' to match export-time inputs; CUDA backend stages CPU input to GPU."
        ),
    )
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.cuda_pte.exists():
        raise FileNotFoundError(f"CUDA model not found: {args.cuda_pte}")
    if not args.cuda_ptd.exists():
        raise FileNotFoundError(f"CUDA external data not found: {args.cuda_ptd}")

    img_bgr = cv2.imread(str(args.image))
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {args.image}")

    yolo = ensure_yolo_predictor(args.ultralytics_model, img_bgr, rect=args.rect)
    input_tensor = preprocess_with_yolo(yolo, img_bgr)
    yolo.predictor.batch = ([str(args.image)], [img_bgr])  # for postprocess path
    print(f"Input tensor shape: {tuple(input_tensor.shape)}")
    if input_tensor.shape[2] != input_tensor.shape[3]:
        print(
            "Warning: non-square preprocessed input detected. "
            "Your current CUDA hybrid export constrains H==W; non-square inputs can trigger CUDA failures."
        )
        if not args.allow_nonsquare_cuda_input:
            raise RuntimeError(
                "Refusing to run CUDA comparison with non-square input. "
                "Use square preprocessing (default: no --rect) or pass --allow-nonsquare-cuda-input."
            )

    xnn_pte = args.xnnpack_pte
    if xnn_pte is None:
        xnn_pte = download_xnnpack_model(args.xnnpack_repo, args.xnnpack_file, args.hf_cache_dir)
    if not xnn_pte.exists():
        raise FileNotFoundError(f"XNNPACK model not found: {xnn_pte}")

    print(f"CUDA model: {args.cuda_pte}")
    print(f"CUDA data: {args.cuda_ptd}")
    print(f"XNNPACK model: {xnn_pte}")

    runtime = Runtime.get()
    print(f"Registered backends: {runtime.backend_registry.registered_backend_names}")

    try:
        cuda_dets = run_executorch_model(
            runtime=runtime,
            pte_path=args.cuda_pte,
            input_tensor=input_tensor,
            predictor=yolo.predictor,
            names=yolo.names,
            conf_threshold=args.conf_threshold,
            packed_format=args.packed_format,
            input_device=args.cuda_input_device,
            ptd_path=args.cuda_ptd,
        )
    except Exception as e:
        raise RuntimeError(f"CUDA model execution failed ({args.cuda_pte}): {e}") from e

    try:
        xnn_dets = run_executorch_model(
            runtime=runtime,
            pte_path=xnn_pte,
            input_tensor=input_tensor,
            predictor=yolo.predictor,
            names=yolo.names,
            conf_threshold=args.conf_threshold,
            packed_format=args.packed_format,
            input_device="cpu",
        )
    except Exception as e:
        raise RuntimeError(f"XNNPACK model execution failed ({xnn_pte}): {e}") from e

    compare_detections("CUDA", cuda_dets, "XNNPACK", xnn_dets, topk=args.topk)
    compare_detections("XNNPACK", xnn_dets, "CUDA", cuda_dets, topk=args.topk)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"Error: {e}")
        raise SystemExit(1)
