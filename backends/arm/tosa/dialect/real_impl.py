# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import logging
from typing import Any, Callable

import numpy as np
import torch
import tosa_reference_model as reference_model  # type: ignore[import-not-found, import-untyped]
import tosa_serializer as ts
from executorch.backends.arm.tosa.mapping import TosaArg
from executorch.backends.arm.tosa.specification import get_context_spec
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode

logger = logging.getLogger(__name__)


def _torch_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.uint16)
    return tensor.numpy()


def _numpy_to_torch_tensor(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    if array.dtype.type is np.void:
        return torch.frombuffer(array, dtype=dtype)

    tensor = torch.from_numpy(array)
    if dtype == torch.bfloat16:
        return tensor.view(torch.bfloat16)
    return tensor


def make_tosa_reference_model_impl(
    fake_func: Callable,
    op_schema: str,
) -> Callable:
    """Create a real eager implementation from a fake TOSA dialect op."""

    signature = inspect.signature(fake_func)
    op_name = op_schema.split("(")[0]

    def real_impl(*args, **kwargs) -> torch.Tensor:

        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        normalized_args = list(bound.arguments.values())

        fake_output = fake_func(*args, **kwargs)
        if not isinstance(fake_output, torch.Tensor):
            raise TypeError(
                f"Only single-tensor outputs are supported for real TOSA op execution, got {type(fake_output).__name__}"
            )

        tensor_args = [arg for arg in normalized_args if isinstance(arg, torch.Tensor)]
        if not tensor_args:
            raise ValueError(
                f"Real TOSA op execution requires at least one tensor input: {op_name}"
            )

        graph = torch.fx.Graph()
        node_args: list[Any] = []
        placeholder_nodes: list[torch.fx.Node] = []
        op_handle = getattr(exir_ops.backend.tosa, op_name).default

        with FakeTensorMode(allow_non_fake_inputs=True) as mode:
            for parameter, arg in zip(signature.parameters.values(), normalized_args):
                if isinstance(arg, torch.Tensor):
                    placeholder = graph.placeholder(parameter.name)
                    placeholder.meta["val"] = mode.from_tensor(arg.detach().cpu())
                    placeholder_nodes.append(placeholder)
                    node_args.append(placeholder)
                else:
                    node_args.append(arg)

            op_node = graph.call_function(op_handle, tuple(node_args), {})
            op_node.meta["val"] = mode.from_tensor(fake_output.detach().cpu())
            graph.output((op_node,))

        tosa_spec = get_context_spec()
        version = tosa_spec.version
        tosa_graph = ts.TosaSerializer(
            "",
            targetMajor=version.major,
            targetMinor=version.minor,
            targetPatch=version.micro,
            targetDraft=False,
        )

        for node in placeholder_nodes:
            arg = TosaArg(node, tosa_spec)
            tosa_graph.addInputTensor(
                ts.TosaSerializerTensor(arg.name, list(arg.shape), arg.dtype, data=None)
            )

        output_arg = TosaArg(op_node, tosa_spec)
        tosa_graph.currRegion.currBasicBlock.addTensor(
            output_arg.name,
            list(output_arg.shape),
            output_arg.dtype,
        )
        from executorch.backends.arm.operators.node_visitor import get_node_visitor

        visitor = get_node_visitor(f"tosa.{op_name}.default", tosa_spec)
        visitor.define_node(
            op_node,
            tosa_graph,
            [TosaArg(arg, tosa_spec) for arg in op_node.args],
            output_arg,
        )
        tosa_graph.addOutputTensor(
            tosa_graph.currRegion.currBasicBlock.tensors[output_arg.name]
        )

        outputs_np, status = reference_model.run(
            tosa_graph.serialize(),
            [_torch_tensor_to_numpy(arg) for arg in tensor_args],
            verbosity=_tosa_refmodel_loglevel(logger.getEffectiveLevel()),
            initialize_variable_tensor_from_numpy=True,
            debug_mode="ALL" if logger.isEnabledFor(logging.DEBUG) else None,
        )
        if status != reference_model.GraphStatus.TOSA_VALID:
            raise RuntimeError(
                f"TOSA reference model rejected tosa.{op_name} graph: {status}"
            )

        return _numpy_to_torch_tensor(outputs_np[0], fake_output.dtype).to(
            device=tensor_args[0].device
        )

    return real_impl


def _tosa_refmodel_loglevel(loglevel: int) -> str:
    """Converts a logging loglevel to tosa_reference_model logginglevel,
    returned as string.
    """
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
