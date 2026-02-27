# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Runner utilities for TensorRT model validation.

Since TensorRT C++ backend isn't registered in Python pybindings,
we run inference via TensorRT Python API for validation.
"""

import logging
from typing import Any, List, Optional, Tuple, Union

import torch
from executorch.exir import EdgeProgramManager

logger = logging.getLogger(__name__)


def _trt_dtype_to_torch(trt_dtype) -> torch.dtype:
    """Convert TensorRT dtype to PyTorch dtype."""
    import tensorrt as trt

    dtype_map = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.bool: torch.bool,
        trt.uint8: torch.uint8,
    }
    # Handle bfloat16 if available (TensorRT 8.6+)
    if hasattr(trt, "bfloat16"):
        dtype_map[trt.bfloat16] = torch.bfloat16

    if trt_dtype not in dtype_map:
        logger.warning(f"Unknown TensorRT dtype {trt_dtype}, defaulting to float32")
        return torch.float32
    return dtype_map[trt_dtype]


def _extract_tensorrt_engine(
    edge_program: EdgeProgramManager,
) -> Optional[bytes]:
    """Extract TensorRT engine from EdgeProgramManager."""
    try:
        from executorch.backends.nvidia.tensorrt.serialization import (
            get_engine_from_blob,
        )

        # Access lowered modules from edge programs
        if not hasattr(edge_program, "_edge_programs"):
            logger.debug("EdgeProgramManager has no _edge_programs attribute")
            return None

        for method_name, edge_prog in edge_program._edge_programs.items():
            if not hasattr(edge_prog, "graph_module"):
                logger.debug(f"Edge program {method_name} has no graph_module")
                continue
            for mod_name, mod in edge_prog.graph_module._modules.items():
                if hasattr(mod, "processed_bytes"):
                    logger.debug(f"Found processed_bytes in {mod_name}")
                    engine = get_engine_from_blob(bytes(mod.processed_bytes))
                    if engine:
                        logger.debug(f"Successfully extracted TensorRT engine ({len(engine)} bytes)")
                        return engine
                    else:
                        logger.debug(f"get_engine_from_blob returned None for {mod_name}")
        logger.debug("No TensorRT engine found in edge program modules")
        return None
    except Exception as e:
        logger.warning(f"Engine extraction failed: {e}")
        return None


def _run_tensorrt_inference(
    engine_bytes: bytes,
    inputs: Tuple[Any, ...],
) -> List[torch.Tensor]:
    """Run inference using TensorRT Python API."""
    import tensorrt as trt

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for TensorRT inference")

    # Deserialize engine
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine")

    context = engine.create_execution_context()

    # Collect I/O info
    input_names, output_info = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            shape = engine.get_tensor_shape(name)
            dtype = engine.get_tensor_dtype(name)
            output_info.append((name, shape, dtype))

    # Prepare inputs
    cuda_inputs = [
        inp.cuda().contiguous() if isinstance(inp, torch.Tensor) else inp
        for inp in inputs
    ]

    # Allocate outputs with correct dtype from engine
    outputs = [
        torch.empty(list(shape), dtype=_trt_dtype_to_torch(dtype), device="cuda")
        for _, shape, dtype in output_info
    ]

    # Bind tensors
    for name, cuda_inp in zip(input_names, cuda_inputs):
        context.set_tensor_address(name, cuda_inp.data_ptr())
    for (name, _, _), output in zip(output_info, outputs):
        context.set_tensor_address(name, output.data_ptr())

    # Execute
    stream = torch.cuda.current_stream().cuda_stream
    if not context.execute_async_v3(stream):
        raise RuntimeError("TensorRT inference failed")
    torch.cuda.synchronize()

    return [out.cpu() for out in outputs]


def run_and_compare(
    reference_model: torch.nn.Module,
    edge_program: EdgeProgramManager,
    sample_inputs: Tuple[Any, ...],
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> bool:
    """Compare TensorRT outputs against PyTorch reference.

    Both reference model and TensorRT run on CUDA for fair comparison.
    Requires a properly configured environment with CUDA-compatible PyTorch.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for TensorRT validation. "
            "Please install PyTorch with CUDA support."
        )

    reference_model.eval()
    reference_model = reference_model.cuda()
    inputs_for_ref = tuple(
        inp.cuda() if isinstance(inp, torch.Tensor) else inp
        for inp in sample_inputs
    )

    with torch.no_grad():
        reference_output = reference_model(*inputs_for_ref)

    if isinstance(reference_output, torch.Tensor):
        reference_output = reference_output.cpu()
    elif isinstance(reference_output, (tuple, list)):
        reference_output = type(reference_output)(
            out.cpu() if isinstance(out, torch.Tensor) else out
            for out in reference_output
        )

    # Try TensorRT Python API
    engine_bytes = _extract_tensorrt_engine(edge_program)
    if engine_bytes is not None:
        try:
            trt_outputs = _run_tensorrt_inference(engine_bytes, sample_inputs)
            return _compare_outputs(reference_output, trt_outputs, atol, rtol)
        except Exception as e:
            logger.warning(f"TensorRT inference failed: {e}")

    # Fallback to pybindings (unlikely to work for TensorRT)
    try:
        from executorch.extension.pybindings.portable_lib import (
            _load_for_executorch_from_buffer,
        )
        from executorch.extension.pytree import tree_flatten

        exec_program = edge_program.to_executorch()
        program_buffer = exec_program.buffer
        et_module = _load_for_executorch_from_buffer(bytes(program_buffer))
        inputs_flat, _ = tree_flatten(sample_inputs)
        et_outputs = et_module.run_method("forward", inputs_flat)
        return _compare_outputs(reference_output, et_outputs, atol, rtol)
    except Exception as e:
        logger.warning(f"Pybindings validation failed: {e}")
        logger.warning(
            "Validation could not be performed. Use C++ runner for full validation."
        )
        return False


def _compare_outputs(
    reference: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    actual: Union[torch.Tensor, List[torch.Tensor]],
    atol: float,
    rtol: float,
) -> bool:
    """Compare outputs within tolerance."""
    ref_list = [reference] if isinstance(reference, torch.Tensor) else list(reference)
    act_list = [actual] if isinstance(actual, torch.Tensor) else list(actual)

    if len(ref_list) != len(act_list):
        logger.error(f"Output count mismatch: {len(ref_list)} vs {len(act_list)}")
        return False

    for i, (ref, act) in enumerate(zip(ref_list, act_list)):
        if not isinstance(ref, torch.Tensor) or not isinstance(act, torch.Tensor):
            continue
        if ref.shape != act.shape:
            logger.error(f"Shape mismatch at {i}: {ref.shape} vs {act.shape}")
            return False
        if not torch.allclose(ref, act, atol=atol, rtol=rtol):
            max_diff = (ref - act).abs().max().item()
            logger.error(f"Value mismatch at {i}: max diff {max_diff:.6f}")
            return False

    return True
