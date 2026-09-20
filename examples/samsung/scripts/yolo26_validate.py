# Copyright (c) Intel Corporation
# Copyright (c) 2026 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# mypy: disable-error-code="import-untyped,import-not-found"

"""
Samsung device test script for YOLO26 model.

Combines model export and device-based validation in a single script call,
following the pattern of run_method_and_compare_outputs from
backends/test/harness/tester.py.

The script:
1. Loads YOLO26 model and preprocesses inputs
2. Exports to .pte for Samsung ENN backend
3. Executes inference on device via RuntimeExecutor (ADB + enn_executor_runner)
4. Validates using the validate_yolo pipeline with device outputs

Usage:
    export EXYNOS_AI_LITECORE_ROOT=/path/to/litecore
    export LD_LIBRARY_PATH=${EXYNOS_AI_LITECORE_ROOT}/lib/x86_64-linux

    # Export and validate on device:
    python test_yolo26.py -c E9955 -m yolo26s -d /path/to/images --validate coco128.yaml 

    # Quantized model with device validation:
    python test_yolo26.py -c E9955 -m yolo26s -d /path/to/images -p A8W8 --validate coco128.yaml 

    A list of available datasets and instructions on how to use a custom dataset can be found at:
    https://docs.ultralytics.com/datasets/detect
    Validation only supports the default --input_dims; please do not specify this parameter when using the
    --validate flag.
"""

import argparse
import glob
import os
from itertools import islice
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from executorch.backends.samsung.partition.enn_partitioner import EnnPartitioner
from executorch.backends.samsung.quantizer import Precision
from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
    PerformanceMode,
)
from executorch.backends.samsung.test.utils.runtime_executor import RuntimeExecutor
from executorch.backends.samsung.test.utils.utils import TestConfig
from executorch.backends.samsung.utils.export_utils import (
    quantize_module,
    to_edge_transform_and_lower_to_enn,
)
from executorch.examples.samsung.utils import save_tensors
from executorch.exir import ExecutorchBackendConfig
from executorch.extension.export_util.utils import save_pte_program
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.validator import BaseValidator as Validator
from ultralytics.utils.torch_utils import unwrap_model


def get_calibration_data_from_folder(
    dataset_path: str,
    transform_fn,
    subset_size: int,
) -> List[Tuple[torch.Tensor, ...]]:
    """Load calibration images from a folder and preprocess them."""
    image_paths = sorted(glob.glob(os.path.join(dataset_path, "*")))
    if not image_paths:
        raise RuntimeError(f"No images found in {dataset_path}")

    calibration_data = []
    for img_path in islice(image_paths, subset_size):
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        input_tensor = transform_fn(frame)
        calibration_data.append((input_tensor,))
    print(f"Loaded {len(calibration_data)} calibration images from {dataset_path}")
    return calibration_data


def _prepare_validation(
    model: YOLO, dataset_yaml_path: str
) -> Tuple[Validator, torch.utils.data.DataLoader]:
    """Prepare the validation pipeline using the ultralytics validator.

    Mirrors the same logic from export_and_validate.py.
    """
    custom = {"rect": False, "batch": 1}
    args = {
        **model.overrides,
        **custom,
        "mode": "val",
    }

    validator = model._smart_load("validator")(args=args, _callbacks=model.callbacks)
    stride = 32
    validator.stride = stride
    validator.data = check_det_dataset(dataset_yaml_path)
    validator.init_metrics(unwrap_model(model))
    validator.device = torch.device("cpu")
    validator.end2end = False

    data_loader = validator.get_dataloader(
        validator.data.get(validator.args.split), validator.args.batch
    )
    return validator, data_loader


def validate_yolo_on_device(
    model: YOLO,
    exec_prog,
    dataset_yaml_path: str,
    pt_model: torch.nn.Module,
    dump: bool = False,
    artifact_dir: str = "./",
) -> Dict[str, float]:
    """Run validation by executing each batch on device and validating with the ultralytics pipeline.

    Mirrors the run_method_and_compare_outputs pattern from the Tester base class:
    1. For each validation batch, preprocess the input
    2. Execute on device via RuntimeExecutor (pushes .pte + input, runs on device, pulls output)
    3. Optionally dump the preprocessed input, CPU reference output, and device output
       each into a per-image directory named after the image file (e.g. 000000000009/)
    4. Postprocess device output using ultralytics validator
    5. Update validation metrics

    Args:
        model: The YOLO model instance.
        exec_prog: The ExecuTorch program manager containing the compiled model.
        dataset_yaml_path: Path to the validation dataset YAML file.
        pt_model: The PyTorch model for computing CPU reference outputs.
        dump: Whether to dump per-batch preprocessed inputs and CPU reference outputs.
        artifact_dir: Directory to save dumped tensors.

    Returns:
        Dictionary of validation statistics computed over the dataset.
    """
    validator, data_loader = _prepare_validation(model, dataset_yaml_path)

    print(f"Start device validation on {dataset_yaml_path} dataset ...")

    batch_idx = 0
    for batch in data_loader:
        img_name = os.path.splitext(os.path.basename(batch["im_file"][0]))[0]
        batch = validator.preprocess(batch)
        input_tensor = batch["img"]

        # Execute on device via RuntimeExecutor
        runtime = RuntimeExecutor(exec_prog, input_tensor)
        device_output = runtime.run_on_device()

        # Optionally dump preprocessed input, CPU reference output, and device output
        if dump:
            batch_dir = os.path.join(artifact_dir, img_name)
            os.makedirs(batch_dir, exist_ok=True)
            save_tensors((input_tensor,), "input", batch_dir)
            with torch.no_grad():
                ref_output = pt_model(input_tensor)
            save_tensors(ref_output, "ref_output", batch_dir)
            save_tensors(device_output, "device_output", batch_dir)
            print(
                f"  Dumped input + ref_output + device_output for {img_name} to {batch_dir}"
            )

        # Postprocess device output and update validation metrics
        device_output_list = (
            list(device_output) if isinstance(device_output, tuple) else [device_output]
        )
        preds = validator.postprocess(device_output_list)
        validator.update_metrics(preds, batch)

        batch_idx += 1
        if batch_idx % 10 == 0:
            print(f"  Processed {batch_idx} batches ...")

    stats = validator.get_stats()
    return stats


def main(args):
    # Load the YOLO model
    print(f"Loading YOLO model: {args.model_name}")
    model = YOLO(args.model_name)

    # Setup preprocessing with target input dimensions
    input_h, input_w = args.input_dims
    np_dummy = np.ones((input_h, input_w, 3), dtype=np.uint8)
    model.predict(np_dummy, imgsz=(input_h, input_w), device="cpu", verbose=False)

    pt_model = model.model.to(torch.device("cpu")).eval()
    float_pt_model = pt_model

    def transform_fn(frame):
        """Preprocess a single frame using the YOLO predictor."""
        return model.predictor.preprocess([frame])

    # Build example input for model export
    example_input = transform_fn(np_dummy)
    example_args = (example_input,)

    # Collect calibration data from image folder
    calibration_number = args.calibration_number
    if args.dataset:
        calibration_data = get_calibration_data_from_folder(
            args.dataset, transform_fn, calibration_number
        )
    else:
        # Use repeated dummy inputs for calibration
        calibration_data = [example_args for _ in range(calibration_number)]
        print(
            f"No dataset provided, using {calibration_number} dummy inputs for calibration."
        )

    # Select test input (first calibration sample)
    test_in = calibration_data[0]
    print(f"Test input shape: {test_in[0].shape}")

    # Compile specs for Samsung ENN backend
    compile_specs = [
        gen_samsung_backend_compile_spec(args.chipset, PerformanceMode.DEFAULT)
    ]

    # Optionally quantize the model
    if args.precision:
        print(f"Quantizing model with precision: {args.precision}")
        pt_model = quantize_module(
            pt_model,
            example_args,
            calibration_data,
            getattr(Precision, args.precision),
        )
        print("Quantization finished.")

    # Lower to Samsung ENN backend and export .pte
    print(f"Lowering model to ENN backend (chipset={args.chipset}) ...")
    edge_prog = to_edge_transform_and_lower_to_enn(
        pt_model, example_args, compile_specs=compile_specs
    )

    edge = edge_prog.to_backend(EnnPartitioner(compile_specs))
    exec_prog = edge.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=True)
    )

    # Save .pte file
    os.makedirs(args.artifact, exist_ok=True)
    pte_filename = f"yolo26_{'int8' if args.precision else 'fp32'}_{args.chipset}"
    save_pte_program(exec_prog, pte_filename, args.artifact)
    pte_path = os.path.join(args.artifact, f"{pte_filename}.pte")
    print(f"Model saved to {pte_path}")

    # Optionally run full validation on device
    if args.validate:
        if args.input_dims != [640, 640]:
            raise NotImplementedError(
                f"Validation with the custom input shape {args.input_dims} is not implemented. "
                "Please use the default --input_dims [640,640] for validation."
            )

        # Configure device connection
        if args.host:
            TestConfig.host_ip = args.host
        if args.device:
            TestConfig.device_id = args.device
        TestConfig.chipset = args.chipset

        print("\nRunning full device validation ...")
        print(f"  Device: {TestConfig.device_id or 'auto-detect'}")
        print(f"  Host: {TestConfig.host_ip or 'localhost'}")
        print(f"  Chipset: {TestConfig.chipset}")
        stats = validate_yolo_on_device(
            model,
            exec_prog,
            args.validate,
            float_pt_model,
            args.dump,
            args.artifact,
        )
        print("Validation results:")
        for stat, value in stats.items():
            print(f"  {stat}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test YOLO26 model on Samsung ENN backend with device validation."
    )

    parser.add_argument(
        "-c",
        "--chipset",
        default="E9955",
        help="Samsung chipset, i.e. E9945, E9955, E9965, etc.",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--model_name",
        default="yolo26n",
        help="Ultralytics YOLO26 model name or path to a .pt file. Default: yolo26n",
        type=str,
    )
    parser.add_argument(
        "--input_dims",
        type=eval,
        default=[640, 640],
        help="Input model dimensions as [height, width]. Default: [640, 640]",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default=None,
        help=(
            "Path to a folder containing calibration images (e.g. COCO images). "
            "Used for calibration."
        ),
        type=str,
    )
    parser.add_argument(
        "-p",
        "--precision",
        default=None,
        choices=[None, "A8W8"],
        help="Quantization precision. If not set, model stays FP32.",
        type=str,
    )
    parser.add_argument(
        "-cn",
        "--calibration_number",
        default=100,
        help="Number of samples for calibrating quantization params. Default: 100.",
        type=int,
    )
    parser.add_argument(
        "--dump",
        default=False,
        const=True,
        nargs="?",
        help="Whether to dump input/output tensors. Default: False.",
        type=bool,
    )
    parser.add_argument(
        "-a",
        "--artifact",
        default="./yolo26",
        help="Path for storing generated artifacts. Default: ./yolo26",
        type=str,
    )
    parser.add_argument(
        "--validate",
        nargs="?",
        const="coco128.yaml",
        help=(
            "Run full validation on device using the Ultralytics validation pipeline. "
            "Provide a path to the dataset YAML file (default: coco128.yaml)."
        ),
        type=str,
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host IP address with device connecting",
        type=str,
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device ID to test",
        type=str,
    )

    args = parser.parse_args()
    main(args)
