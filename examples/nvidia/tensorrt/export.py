# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Export models to ExecuTorch format with the TensorRT delegate.

Usage:
    python -m executorch.examples.nvidia.tensorrt.export -m add
    python -m executorch.examples.nvidia.tensorrt.export -o ./output
"""

import argparse
import logging
from typing import List

import torch
from torch.export import export
from torch.utils._pytree import tree_flatten

from executorch.backends.nvidia.tensorrt.partitioner import TensorRTPartitioner
from executorch.examples.models import MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


# Models that are supported by TensorRT delegate (sorted alphabetically).
TENSORRT_SUPPORTED_MODELS = {
    "add",
    "add_mul",
    "mul",
}


def get_supported_models_list() -> list:
    """Return list of models supported by TensorRT backend."""
    return sorted(TENSORRT_SUPPORTED_MODELS)


def export_model(
    model_name: str,
    output_dir: str,
    strict: bool,
    logger: logging.Logger,
) -> tuple:
    """Export a single model to ExecuTorch format with TensorRT delegate.

    Returns:
        (model, example_inputs, exec_prog) for optional downstream use.
    """
    if model_name not in MODEL_NAME_TO_MODEL:
        raise ValueError(
            f"Model '{model_name}' is not registered in MODEL_NAME_TO_MODEL. "
            f"Available: {list(MODEL_NAME_TO_MODEL.keys())}"
        )

    logging.info(f"Creating model: {model_name}")
    torch.manual_seed(0)
    model, example_inputs, _, dynamic_shapes = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[model_name]
    )

    model.eval()

    logging.info(f"Exporting model {model_name} with TensorRT delegate")

    if dynamic_shapes is not None:
        program = export(
            model, example_inputs, dynamic_shapes=dynamic_shapes, strict=strict
        )
    else:
        program = export(model, example_inputs, strict=strict)

    edge_program = to_edge_transform_and_lower(
        program,
        partitioner=[TensorRTPartitioner()],
    )

    logging.info(f"Lowered graph:\n{edge_program.exported_program().graph}")

    exec_prog = edge_program.to_executorch()

    output_filename = f"{model_name}_tensorrt"
    save_pte_program(exec_prog, output_filename, output_dir)
    logger.info(f"Model exported and saved as {output_filename}.pte in {output_dir}")

    return model, example_inputs, exec_prog


# ---------------------------------------------------------------------------
# Correctness verification (used by test_export.py via buck test)
# ---------------------------------------------------------------------------

_TEST_SEEDS = [2025, 12345, 7777]
_ATOL = 1e-3
_RTOL = 1e-3


def _verify_correctness(
    model_name: str,
    model: torch.nn.Module,
    example_inputs: tuple,
    pte_bytes: bytes,
    logger: logging.Logger,
) -> None:
    """Run exported program with random inputs, compare to eager on GPU."""
    from executorch.extension.pybindings.portable_lib import (
        _load_for_executorch_from_buffer,
    )

    et_module = _load_for_executorch_from_buffer(pte_bytes)

    for seed in _TEST_SEEDS:
        inputs = _randomise_inputs(example_inputs, seed)

        eager_outputs = _run_eager(model, inputs)
        et_outputs = _run_executorch(et_module, inputs)

        n = min(len(eager_outputs), len(et_outputs))
        if n == 0:
            logger.warning(f"{model_name} seed={seed}: no comparable outputs")
            continue

        max_diff = 0.0
        for i in range(n):
            e, a = eager_outputs[i], et_outputs[i]
            if e.shape != a.shape:
                continue
            if e.dtype in (torch.int32, torch.int64, torch.long):
                continue
            if torch.isnan(e).any() or torch.isnan(a).any():
                continue
            diff = (e.float() - a.float()).abs().max().item()
            max_diff = max(max_diff, diff)
            if not torch.allclose(e.float(), a.float(), atol=_ATOL, rtol=_RTOL):
                raise RuntimeError(
                    f"FAIL: {model_name} output[{i}] seed={seed}: "
                    f"max diff {diff:.6e} exceeds tolerance "
                    f"(atol={_ATOL}, rtol={_RTOL})"
                )

        logger.info(
            f"PASS: {model_name} seed={seed} "
            f"({n} outputs, max_diff={max_diff:.6e})"
        )


def _randomise_inputs(example_inputs: tuple, seed: int) -> tuple:
    """Create inputs with small perturbation for numerical diversity."""
    torch.manual_seed(seed)

    def _perturb(obj):
        if isinstance(obj, torch.Tensor):
            if obj.dtype in (torch.int32, torch.int64, torch.long):
                return obj.clone()
            if obj.dtype == torch.bool:
                return torch.randint(0, 2, obj.shape).bool()
            return obj + torch.rand_like(obj) * 0.002 - 0.001
        if isinstance(obj, (list, tuple)):
            return type(obj)(_perturb(x) for x in obj)
        return obj

    return tuple(_perturb(inp) for inp in example_inputs)


def _run_eager(
    model: torch.nn.Module, inputs: tuple
) -> List[torch.Tensor]:
    """Run eager PyTorch on GPU with TF32 disabled for strict FP32 reference."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = model.cuda().eval()

    def _to_cuda(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cuda()
        elif isinstance(obj, (list, tuple)):
            return type(obj)(_to_cuda(item) for item in obj)
        else:
            return obj

    cuda_inputs = tuple(_to_cuda(inp) for inp in inputs)
    with torch.no_grad():
        out = model(*cuda_inputs)
    model.cpu()
    return [t.cpu() for t in tree_flatten(out)[0] if isinstance(t, torch.Tensor)]


def _run_executorch(module, inputs: tuple) -> List[torch.Tensor]:
    """Run inference via ExecuTorch pybindings."""
    flat, _ = tree_flatten(inputs)
    flat = [x for x in flat if isinstance(x, torch.Tensor) or x is None]
    outputs = module.forward(flat)
    return [t for t in outputs if isinstance(t, torch.Tensor)]


def main() -> None:
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Export models to ExecuTorch format with TensorRT delegate"
    )
    parser.add_argument(
        "-m",
        "--model_name",
        required=False,
        help=f"Model name to export (default: all). Supported: {get_supported_models_list()}",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=".",
        help="Output directory for exported model (default: current directory)",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        default=True,
        help="Disable strict mode for export (default: strict mode enabled)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    models = [args.model_name] if args.model_name else sorted(TENSORRT_SUPPORTED_MODELS)
    failed = []
    for model_name in models:
        try:
            export_model(model_name, args.output_dir, args.strict, logger)
        except Exception as e:
            logging.error(f"Failed to export {model_name}: {e}")
            failed.append(model_name)
    if failed:
        logging.error(f"Failed models: {failed}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
