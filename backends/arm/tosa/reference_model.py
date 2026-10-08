# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Execute Arm TOSA delegates with the TOSA reference model."""

from __future__ import annotations

import json
import logging
import numbers
import subprocess  # nosec B404 - invoked only for the configured tool binary
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm.constants import (
    NHWC_INVERSE_ORDER,
    NHWC_ORDER,
    NNHWC_INVERSE_ORDER,
    NNHWC_ORDER,
)
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.tosa.specification import Tosa_1_00, TosaSpecification
from executorch.exir import ExportedProgram
from executorch.exir.lowered_backend_module import LoweredBackendModule
from torch.fx.node import Node
from torch.overrides import TorchFunctionMode

logger: logging.Logger = logging.getLogger(__name__)

INFER_SHAPES_PATH = "infer_shapes"

_QDQ_TORCH_OVERLOADS = (
    ("quantize_per_tensor", ("tensor", "tensor2", "default")),
    ("dequantize_per_tensor", ("tensor", "tensor2", "default")),
    ("quantize_per_channel", ("default",)),
    ("dequantize_per_channel", ("default",)),
)

_QDQ_BACKWARD_COMPAT_OVERLOADS = (
    ("quantize_per_tensor", ("out",)),
    ("dequantize_per_tensor", ("out",)),
    ("quantize_per_channel", ("out",)),
    ("dequantize_per_channel", ("out",)),
)


def _get_qdq_memory_format_ops() -> tuple[object, ...]:
    qdq_ops = []
    backward_compat = dict(_QDQ_BACKWARD_COMPAT_OVERLOADS)
    namespace = torch.ops.quantized_decomposed
    for op_name, overload_names in _QDQ_TORCH_OVERLOADS:
        op_packet = getattr(namespace, op_name, None)
        if op_packet is None:
            continue
        for overload_name in overload_names + backward_compat[op_name]:
            if hasattr(op_packet, overload_name):
                qdq_ops.append(getattr(op_packet, overload_name))
    return tuple(qdq_ops)


_QDQ_MEMORY_FORMAT_OPS = _get_qdq_memory_format_ops()


def get_input_names(
    program: ExportedProgram, is_lowered_module: bool = False
) -> list[str]:
    """Return model inputs in graph-signature order."""
    if not is_lowered_module:
        return [spec.arg.name for spec in program.graph_signature.input_specs]
    return [
        user_input
        for user_input in program.graph_signature.user_inputs
        if isinstance(user_input, str)
    ]


def torch_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    dim_order = tensor.dim_order()
    if dim_order == NHWC_ORDER:
        tensor = tensor.permute(NHWC_ORDER)
    elif dim_order == NNHWC_ORDER:
        tensor = tensor.permute(NNHWC_ORDER)

    tensor = tensor.detach()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.uint16)
    elif tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        tensor = tensor.view(torch.uint8)
    return tensor.numpy()


def torch_tensor_to_tosa_shape(tensor: torch.Tensor) -> list[int]:
    shape = list(tensor.shape)
    dim_order = tensor.dim_order()
    if dim_order in (NHWC_ORDER, NNHWC_ORDER):
        shape = [shape[index] for index in dim_order]
    return [int(dim) for dim in shape]


def user_inputs_need_shape_inference(program: ExportedProgram) -> bool:
    user_inputs = {
        user_input
        for user_input in program.graph_signature.user_inputs
        if isinstance(user_input, str)
    }
    for node in program.graph.nodes:
        if node.op != "placeholder" or node.name not in user_inputs:
            continue
        input_tensor = get_first_fake_tensor(node)
        if any(not isinstance(dim, numbers.Integral) for dim in input_tensor.shape):
            return True
    return False


def numpy_to_torch_tensor(array: np.ndarray, output_node: Node) -> torch.Tensor:
    output_tensor = get_first_fake_tensor(output_node)
    shape = output_tensor.shape
    dim_order = output_tensor.dim_order()

    def is_concrete_shape(shape_like) -> bool:
        return all(isinstance(dim, numbers.Integral) for dim in shape_like)

    def to_torch_tensor() -> torch.Tensor:
        if output_tensor.dtype == torch.bfloat16 or array.dtype.type is np.void:
            return torch.frombuffer(array, dtype=output_tensor.dtype)
        return torch.from_numpy(array)

    if dim_order == NHWC_ORDER:
        tensor = to_torch_tensor()
        if is_concrete_shape(shape):
            tensor = tensor.reshape([shape[index] for index in NHWC_ORDER])
        return tensor.permute(NHWC_INVERSE_ORDER).contiguous()
    if dim_order == NNHWC_ORDER:
        tensor = to_torch_tensor()
        if is_concrete_shape(shape):
            tensor = tensor.reshape([shape[index] for index in NNHWC_ORDER])
        return tensor.permute(NNHWC_INVERSE_ORDER).contiguous()

    tensor = to_torch_tensor()
    if is_concrete_shape(shape):
        return tensor.reshape(shape)
    return tensor


def _tosa_refmodel_loglevel(loglevel: int) -> str:
    """Convert a Python logging level to a TOSA reference-model level."""
    loglevel_map = {
        logging.INFO: "INFO",
        logging.CRITICAL: "LOW",
        logging.ERROR: "LOW",
        logging.WARNING: "MED",
        logging.DEBUG: "HIGH",
        logging.NOTSET: "MED",
    }
    clamped_logging_level = max(min(loglevel // 10 * 10, 50), 0)
    return loglevel_map[clamped_logging_level]


def run_tosa_graph(
    graph: Any,
    tosa_version: TosaSpecification,
    inputs: Sequence[torch.Tensor],
    output_node: Node,
) -> list[torch.Tensor]:
    """Run a serialized TOSA graph and restore declared output layouts."""
    inputs_np = [torch_tensor_to_numpy(input_tensor) for input_tensor in inputs]

    if not isinstance(tosa_version, Tosa_1_00):
        raise ValueError(
            f"Unknown TOSA specification: {tosa_version}. No reference model "
            "is available for this specification version"
        )

    import tosa_reference_model as reference_model  # type: ignore[import-not-found, import-untyped]

    outputs_np, status = reference_model.run(
        graph,
        inputs_np,
        verbosity=_tosa_refmodel_loglevel(logger.getEffectiveLevel()),
        initialize_variable_tensor_from_numpy=True,
        debug_mode="ALL" if logger.getEffectiveLevel() <= logging.DEBUG else None,
    )
    if status != reference_model.GraphStatus.TOSA_VALID:
        raise RuntimeError("Non-valid TOSA given to reference model.")
    return [
        numpy_to_torch_tensor(output_array, node)
        for output_array, node in zip(outputs_np, output_node.args[0])  # type: ignore[arg-type]
    ]


class TosaReferenceModelDispatch(TorchFunctionMode):
    """Execute ``TOSABackend`` delegates through the TOSA reference model."""

    def __init__(self) -> None:
        self.ran_tosa_dispatch = False
        super().__init__()

    def _generate_shape_inference_json(
        self,
        tosa_buffer: bytes,
        artifact_path: Path,
        test_case_path: Path,
        input_names: list[str],
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        shapes = dict(
            zip(input_names, [torch_tensor_to_tosa_shape(value) for value in inputs])
        )
        with test_case_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {"tosa_file": str(artifact_path), "shapes": shapes}, handle, indent=2
            )

    def _run_infer_shapes(
        self,
        tosa_buffer: bytes,
        input_names: list[str],
        inputs: tuple[torch.Tensor, ...],
        temp_dir_path: Path,
        infer_shapes_path: str = INFER_SHAPES_PATH,
    ) -> bytes:
        model_suffix = "model.tosa"
        tosa_sym_int_model = temp_dir_path / model_suffix
        tosa_sym_int_model.write_bytes(tosa_buffer)
        test_case_file = temp_dir_path / "test_case.json"

        self._generate_shape_inference_json(
            tosa_buffer, tosa_sym_int_model, test_case_file, input_names, inputs
        )
        subprocess.run(
            [infer_shapes_path, str(test_case_file)],
            check=True,
            capture_output=True,
            text=True,
        )  # nosec
        resolved_file = temp_dir_path / f"resolved_{model_suffix}"
        return resolved_file.read_bytes()

    def _tosa_dispatch(self, lowered_backend_module: LoweredBackendModule, inputs):
        tosa_buffer = lowered_backend_module.processed_bytes
        compile_spec = TosaCompileSpec._from_list(lowered_backend_module.compile_specs)
        tosa_spec = compile_spec.tosa_spec
        output_node = lowered_backend_module.original_module.graph.output_node()
        if tosa_spec.support_extension("shape") and user_inputs_need_shape_inference(
            lowered_backend_module.original_module
        ):
            input_names = get_input_names(lowered_backend_module.original_module, True)
            with tempfile.TemporaryDirectory() as temp_dir:
                tosa_buffer = self._run_infer_shapes(
                    tosa_buffer,
                    input_names,
                    inputs,
                    Path(temp_dir),
                )

        return run_tosa_graph(tosa_buffer, tosa_spec, inputs, output_node)

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        # Only raise this error if we ran the model without errors.
        if not self.ran_tosa_dispatch and exc_type is None:
            raise RuntimeError(
                "Ran model with TosaReferenceModelDispatch but never ran TOSABackend delegate."
            )

    def __torch_function__(self, func, types, args=..., kwargs=None):
        if func is torch._higher_order_ops.executorch_call_delegate:
            lowered_backend_module = cast(LoweredBackendModule, args[0])
            if lowered_backend_module.backend_id == "TOSABackend":
                self.ran_tosa_dispatch = True
                return self._tosa_dispatch(lowered_backend_module, args[1:])
            else:
                raise RuntimeError(
                    f"Ran model with TosaReferenceModelDispatch but call_delegate with {lowered_backend_module.backend_id=} != 'TOSABackend'."
                )

        kwargs = kwargs or {}
        if func in _QDQ_MEMORY_FORMAT_OPS:
            input_dim_order = args[0].dim_order()
            if input_dim_order in (NHWC_ORDER, NNHWC_ORDER):
                args = [args[0].to(memory_format=torch.contiguous_format), *args[1:]]
                res = func(*args, **kwargs)
                return res.to(memory_format=torch.channels_last)

        return func(*args, **kwargs)
