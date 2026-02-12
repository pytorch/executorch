#!/usr/bin/env python3
# Export YOLO26 with hybrid CPU/CUDA + dynamic shapes support

# CRITICAL: Set multiprocessing to 'spawn' before any imports
# This must happen before torch/CUDA initialization
import multiprocessing
import os

# Set spawn method (CUDA-safe)
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Disable worker pool so compilation happens in main process with GPU access
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import Dim
from ultralytics import YOLO


MODEL_SIZES = ["n", "s", "m", "l", "x"]
TASK_TYPES = ["", "-seg", "-pose", "-obb", "-cls"]

EDGE_COMPILE_CONFIG = EdgeCompileConfig(
    _check_ir_validity=False,
    _skip_dim_order=True,
)

_INDUCTOR_LOWERINGS_REGISTERED = False

torch.set_float32_matmul_precision('high')

def register_inductor_lowerings() -> None:
    """Register custom Inductor lowerings for ops that lack AOTI c-shim support."""
    global _INDUCTOR_LOWERINGS_REGISTERED
    if _INDUCTOR_LOWERINGS_REGISTERED:
        return

    from torch._inductor import lowering as L
    from torch._decomp import get_decompositions

    @L.register_lowering(torch.ops.aten.detach_copy.default, type_promotion_kind=None)
    def detach_copy_lowering(x):
        return L.clone(x)

    @L.register_lowering(torch.ops.aten.split_copy.Tensor, type_promotion_kind=None)
    def split_copy_tensor_lowering(x, split_size, dim=0):
        return [L.clone(chunk) for chunk in L.split(x, split_size, dim)]

    # Register decomposition for transposed convolution (used in segmentation models)
    # This decomposes conv_transpose2d into operations Triton can handle
    decomps = get_decompositions([torch.ops.aten.conv_transpose2d])
    for op, decomp_fn in decomps.items():
        L.register_lowering(op)(decomp_fn)

    print(f"  Registered custom lowerings for detach_copy, split_copy, and conv_transpose2d")
    _INDUCTOR_LOWERINGS_REGISTERED = True


def get_all_model_variants() -> list[str]:
    """Generate all YOLO26 model variant names."""
    variants = []
    for size in MODEL_SIZES:
        for task in TASK_TYPES:
            variants.append(f"yolo26{size}{task}")
    return variants


def create_test_image(task_type: str) -> np.ndarray:
    """Create test image for different task types."""
    width, height = 640, 640
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)

    if task_type == "-cls":
        cv2.putText(img, "Test", (250, 320), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    else:
        cv2.rectangle(img, (100, 150), (180, 400), (100, 150, 100), -1)
        cv2.circle(img, (140, 150), 40, (150, 120, 100), -1)
        cv2.rectangle(img, (300, 250), (500, 400), (150, 50, 50), -1)

    return img


class YOLOBackboneOnly(torch.nn.Module):
    """
    Wrapper that extracts only backbone and head raw outputs.
    Excludes postprocessing (NMS, topk) which runs on CPU.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        print(f"  Using backbone-only mode (no postprocessing)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.export = True
        output = self.model(x)
        if isinstance(output, (list, tuple)):
            return output[0]
        return output


class YOLOWithPostprocess(torch.nn.Module):
    """
    Full YOLO pipeline: backbone + head + postprocessing (topk/gather).
    Outputs packed detections [B, max_det, 6] with columns [x, y, w, h, conf, cls].
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

        for m in self.model.modules():
            if type(m).__name__ in ['Detect', 'Segment', 'Pose', 'OBB', 'Classify']:
                m.export = True
                m.agnostic_nms = True
                print(f"  Set export=True, agnostic_nms=True on {type(m).__name__} head")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if isinstance(output, (list, tuple)):
            return output[0]
        return output


def export_model_variant(
    model_name: str,
    base_path: Path,
    test_model: bool = True,
    skip_existing: bool = False,
    dynamic_shapes: bool = False,
    min_size: int = 320,
    max_size: int = 1280,
    include_postprocess: bool = True,
) -> Tuple[bool, str]:
    """Export YOLO26 model with CUDA backend and optional dynamic shapes."""
    variant_dir = base_path / model_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    model_pte_path = variant_dir / "model.pte"
    data_path = variant_dir / "aoti_cuda_blob.ptd"

    if skip_existing and model_pte_path.exists() and data_path.exists():
        return True, "Skipped (already exists)"

    try:
        print(f"\n{'='*70}")
        shape_info = "dynamic" if dynamic_shapes else "static"
        mode_str = "full model" if include_postprocess else "backbone only"
        print(f"Exporting {model_name} ({mode_str}, {shape_info} shapes)")
        print(f"{'='*70}")

        print(f"Loading model from Ultralytics...")
        model = YOLO(f"{model_name}.pt")

        input_dims = (640, 640)
        task_type = ""
        if "-seg" in model_name:
            task_type = "-seg"
        elif "-pose" in model_name:
            task_type = "-pose"
        elif "-obb" in model_name:
            task_type = "-obb"
        elif "-cls" in model_name:
            task_type = "-cls"

        dummy_img = create_test_image(task_type)

        print(f"Preparing model...")
        model.predict(dummy_img, imgsz=input_dims, device="cpu")
        pt_model = model.model.to(torch.device("cpu"))
        pt_model.eval()

        # Monkey-patch get_topk_index to fix float32 index issue (CUDA AOTI compatible)
        def fixed_get_topk_index(self, scores: torch.Tensor, max_det: int):
            """Get top-k indices with int64 class indices (CUDA AOTI compatible)."""
            batch_size, anchors, nc = scores.shape
            k = max_det if self.export else min(max_det, anchors)

            if self.agnostic_nms:
                scores_max, labels = scores.max(dim=-1, keepdim=True)
                scores_out, indices = scores_max.topk(k, dim=1)
                labels = labels.gather(1, indices)
                return scores_out, labels, indices

            # Non-agnostic NMS path
            ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)  # [batch, k, 1]
            scores_gathered = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
            scores_out, index = scores_gathered.flatten(1).topk(k)  # index: [batch, k]

            # Ensure index is int64
            index = index.to(torch.int64)

            # Compute class indices (modulo operation) - ensure int64
            class_indices = torch.fmod(index, nc).to(torch.int64).unsqueeze(-1)  # [batch, k, 1]

            # Compute anchor indices using floor_divide - ensure int64
            anchor_idx = torch.div(index, nc, rounding_mode='trunc').to(torch.int64).unsqueeze(-1)  # [batch, k, 1]
            idx = torch.gather(ori_index, dim=1, index=anchor_idx)  # [batch, k, 1]

            return scores_out.unsqueeze(-1), class_indices, idx

        # Apply the fix to all detection heads (YOLO26 uses Detect26, Segment26, etc.)
        for m in pt_model.modules():
            module_name = type(m).__name__
            if any(x in module_name for x in ['Detect', 'Segment', 'Pose', 'OBB', 'Classify']):
                if hasattr(m, 'get_topk_index'):
                    m.get_topk_index = fixed_get_topk_index.__get__(m, type(m))
                    print(f"  Applied get_topk_index fix to {module_name} head")

        # Replace ConvTranspose2d with decomposed version for segmentation models
        if task_type == "-seg":
            print(f"  Replacing ConvTranspose2d with UpsampleConv for CUDA compatibility...")
            replaced_count = 0
            for name, module in list(pt_model.named_modules()):
                if isinstance(module, torch.nn.ConvTranspose2d):
                    # Create replacement: Upsample + Conv2d (mathematically equivalent for stride=2)
                    if module.stride == (2, 2) and module.kernel_size == (2, 2):
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]
                        parent = pt_model
                        for part in parent_name.split('.'):
                            if part:
                                parent = getattr(parent, part)

                        # Upsample + Conv2d decomposition
                        class UpsampleConv(torch.nn.Module):
                            def __init__(self, conv_t):
                                super().__init__()
                                self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
                                # Convert ConvTranspose2d weights to Conv2d
                                self.conv = torch.nn.Conv2d(
                                    conv_t.in_channels, conv_t.out_channels,
                                    kernel_size=3, stride=1, padding=1,
                                    bias=(conv_t.bias is not None)
                                )
                                self.conv.weight.data = conv_t.weight.data
                                if conv_t.bias is not None:
                                    self.conv.bias.data = conv_t.bias.data

                            def forward(self, x):
                                return self.conv(self.upsample(x))

                        setattr(parent, child_name, UpsampleConv(module))
                        replaced_count += 1

            if replaced_count > 0:
                print(f"  Replaced {replaced_count} ConvTranspose2d layer(s)")

        if include_postprocess:
            wrapped_model = YOLOWithPostprocess(pt_model)
        else:
            wrapped_model = YOLOBackboneOnly(pt_model)
        wrapped_model.eval()

        def transform_fn(frame):
            return model.predictor.preprocess([frame])

        example_input = transform_fn(dummy_img)
        print(f"Input shape: {example_input.shape}")

        # Set up dynamic shapes if requested
        dynamic_shapes_spec = None
        if dynamic_shapes:
            # For YOLO: dynamic height/width (multiples of 32), static batch=1 and channels=3
            height_dim = Dim("height", min=min_size, max=max_size)
            width_dim = Dim("width", min=min_size, max=max_size)

            # Input is (batch, channels, height, width)
            dynamic_shapes_spec = {
                "x": {
                    0: None,  # batch: static at 1
                    1: None,  # channels: static at 3
                    2: height_dim,  # height: dynamic
                    3: width_dim,   # width: dynamic
                }
            }

            print(f"Dynamic shapes configuration:")
            print(f"  Batch: 1 (static)")
            print(f"  Channels: 3 (static)")
            print(f"  Height: {min_size}-{max_size} (dynamic, multiple of 32)")
            print(f"  Width: {min_size}-{max_size} (dynamic, multiple of 32)")

        print(f"\nExporting to ATEN dialect...")
        with torch.no_grad():
            exported_program = torch.export.export(
                wrapped_model,
                args=(example_input,),
                dynamic_shapes=dynamic_shapes_spec,
                strict=False,  # Non-strict mode (recommended)
            )

        # Check what conv operations are in the graph before decomposition
        print(f"  Checking operations before decomposition...")
        conv_ops_before = []
        conv_transpose_ops = []
        for node in exported_program.graph.nodes:
            if node.op == 'call_function' and 'conv' in str(node.target).lower():
                conv_ops_before.append((node.name, str(node.target), len(node.args) > 6 and node.args[6]))
                if 'transpose' in str(node.target).lower():
                    conv_transpose_ops.append((node.name, str(node.target)))
        print(f"  Found {len(conv_ops_before)} conv operations, {len(conv_transpose_ops)} are conv_transpose")
        if conv_transpose_ops:
            for name, target in conv_transpose_ops:
                print(f"    {name}: {target}")

        # Apply decompositions for transposed convolution (needed for segmentation models)
        from torch._inductor.decomposition import core_aten_decompositions
        decomp_table = core_aten_decompositions()

        # Verify conv_transpose2d decomposition is present
        has_conv_transpose_decomp = any('conv_transpose2d' in str(op) for op in decomp_table.keys())
        print(f"  Decomposition table has conv_transpose2d: {has_conv_transpose_decomp}")

        print(f"  Applying decompositions for conv_transpose2d...")
        exported_program = exported_program.run_decompositions(decomp_table)

        # Check operators
        ops_used = set()
        transposed_convs = []
        for node in exported_program.graph.nodes:
            if node.op == 'call_function':
                ops_used.add(str(node.target))
                # Check for transposed convolutions
                if 'conv' in str(node.target).lower():
                    # Check if this is a transposed convolution
                    if len(node.args) > 6:
                        transposed_arg = node.args[6]
                        if transposed_arg is True or (hasattr(transposed_arg, 'name') and 'True' in str(transposed_arg)):
                            transposed_convs.append(node.name)

        print(f"Operators in graph: {len(ops_used)}")

        # Check if conv_transpose2d was decomposed
        has_conv_transpose = any('conv_transpose' in str(op) for op in ops_used) or len(transposed_convs) > 0
        if has_conv_transpose:
            print(f"  ⚠️  conv_transpose2d still in graph - decomposition may not have worked")
            if transposed_convs:
                print(f"     Found {len(transposed_convs)} transposed convolution nodes: {transposed_convs[:3]}")
        else:
            print(f"  ✓ conv_transpose2d decomposed successfully")

        problematic_ops = ['index_put', 'topk', 'index.Tensor']
        found_problematic = [op for op in problematic_ops if any(op in str(used_op) for used_op in ops_used)]

        if found_problematic:
            print(f"⚠️  Warning: Potentially problematic operators: {found_problematic}")
        else:
            print(f"✓ No problematic operators found")

        # Register custom lowerings for missing ops
        register_inductor_lowerings()

        print(f"\nLowering to CUDA backend...")

        partitioner = CudaPartitioner(
            [CudaBackend.generate_method_name_compile_spec(model_name)]
        )

        et_prog = to_edge_transform_and_lower(
            exported_program,
            partitioner=[partitioner],
            compile_config=EDGE_COMPILE_CONFIG,
        )

        print(f"Converting to ExecuTorch program...")
        exec_program = et_prog.to_executorch()

        print(f"Saving model...")
        with open(model_pte_path, "wb") as f:
            exec_program.write_to_file(f)

        print(f"Saved:")
        print(f"  - {model_pte_path} ({model_pte_path.stat().st_size / 1024 / 1024:.2f} MB)")
        if data_path.exists():
            print(f"  - {data_path} ({data_path.stat().st_size / 1024 / 1024:.2f} MB)")

        if test_model:
            print(f"\nTesting inference...")
            test_success, test_msg = test_model_inference(
                model_pte_path, data_path, example_input, dynamic_shapes, min_size, max_size
            )
            if not test_success:
                return False, f"Export OK, test failed: {test_msg}"
            print(f"✓ {test_msg}")

        return True, "Success"

    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, f"Failed: {str(e)}"


def test_model_inference(
    model_path: Path,
    data_path: Path,
    example_input: torch.Tensor,
    dynamic_shapes: bool,
    min_size: int,
    max_size: int,
) -> Tuple[bool, str]:
    """Test inference on exported CUDA model."""
    try:
        if not torch.cuda.is_available():
            return False, "CUDA not available"

        from executorch.runtime import Runtime

        runtime = Runtime.get()

        with open(model_path, "rb") as f:
            program = runtime.load_program(f.read())

        method = program.load_method("forward")
        if method is None:
            return False, "Failed to load method"

        # Test with original input
        outputs = method.execute((example_input.contiguous(),))

        if outputs is None or len(outputs) == 0:
            return False, "No outputs"

        output_shape = outputs[0].shape
        msg = f"Test passed, output: {output_shape}"

        # If dynamic shapes, test with different size
        if dynamic_shapes and max_size > min_size:
            alt_size = min_size if example_input.shape[2] == max_size else max_size
            alt_input = torch.randn(1, 3, alt_size, alt_size)
            outputs2 = method.execute((alt_input.contiguous(),))

            if outputs2 is None or len(outputs2) == 0:
                return False, f"Alt size {alt_size} failed"

            msg += f", alt {alt_size}x{alt_size}: {outputs2[0].shape}"

        return True, msg

    except Exception as e:
        return False, f"Inference failed: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLO26 with CUDA (hybrid CPU/GPU + dynamic shapes)"
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        required=True,
        help="Base directory for exports",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Specific models to export (default: all)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=True,
        help="Test after export (default: True)",
    )
    parser.add_argument(
        "--no-test",
        action="store_false",
        dest="test",
        help="Skip testing",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if already exported",
    )
    parser.add_argument(
        "--dynamic-shapes",
        action="store_true",
        help="Enable dynamic height/width (variable input resolutions)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=320,
        help="Min dimension (multiple of 32, default: 320)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1280,
        help="Max dimension (multiple of 32, default: 1280)",
    )
    parser.add_argument(
        "--backbone-only",
        action="store_true",
        help="Export backbone only (no postprocessing). Default exports full model with postprocessing.",
    )

    args = parser.parse_args()

    # Validate
    if args.min_size % 32 != 0:
        print(f"Error: min-size must be multiple of 32")
        return 1
    if args.max_size % 32 != 0:
        print(f"Error: max-size must be multiple of 32")
        return 1
    if args.min_size > args.max_size:
        print(f"Error: min-size > max-size")
        return 1

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available")

    args.base_path.mkdir(parents=True, exist_ok=True)

    model_variants = args.models if args.models else get_all_model_variants()

    print(f"\n{'='*70}")
    print(f"YOLO26 CUDA Export")
    print(f"{'='*70}")
    mode_str = "Backbone only" if args.backbone_only else "Full model with postprocessing"
    print(f"Mode: {mode_str}")
    print(f"Base path: {args.base_path.absolute()}")
    print(f"Models: {len(model_variants)}")
    print(f"Dynamic shapes: {args.dynamic_shapes}")
    if args.dynamic_shapes:
        print(f"  Range: {args.min_size}-{args.max_size} (multiples of 32)")

    results = {}

    for i, model_name in enumerate(model_variants, 1):
        print(f"\n[{i}/{len(model_variants)}] {model_name}...")

        success, message = export_model_variant(
            model_name,
            args.base_path,
            test_model=args.test,
            skip_existing=args.skip_existing,
            dynamic_shapes=args.dynamic_shapes,
            min_size=args.min_size,
            max_size=args.max_size,
            include_postprocess=not args.backbone_only,
        )

        results[model_name] = (success, message)

    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")

    successful = sum(1 for s, _ in results.values() if s)
    failed = len(results) - successful

    print(f"\nTotal: {len(results)} | Success: {successful} | Failed: {failed}\n")

    if successful > 0:
        print("✓ Successful:")
        for model_name, (success, msg) in results.items():
            if success:
                print(f"  {model_name}: {msg}")

    if failed > 0:
        print("\n✗ Failed:")
        for model_name, (success, msg) in results.items():
            if not success:
                print(f"  {model_name}: {msg}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
