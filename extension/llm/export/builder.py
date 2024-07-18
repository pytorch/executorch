# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Providing builders for LLM models. These builders help user to build LLM
# eager models, apply source transformations and quantization and export them to
# ExecuTorch.

import logging
from enum import Enum
from typing import Any, Callable, List, Optional

import torch
from executorch.backends.transforms.duplicate_dynamic_quant_chain import (
    DuplicateDynamicQuantChainPass,
)
from executorch.exir import EdgeProgramManager
from executorch.exir.backend.partitioner import Partitioner

from executorch.exir.backend.utils import format_delegated_graph
from executorch.exir.capture._config import EdgeCompileConfig, ExecutorchBackendConfig

from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass

from executorch.extension.export_util.utils import export_to_edge, save_pte_program
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer import Quantizer
from torch.ao.quantization.quantizer.composable_quantizer import ComposableQuantizer
from torch.nn.attention import SDPBackend

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


class DType(Enum):
    fp32 = "fp32"
    fp16 = "fp16"
    bf16 = "bf16"

    def to_torch_dtype(self) -> torch.dtype:
        mapping = {
            DType.fp32: torch.float32,
            DType.fp16: torch.float16,
        }
        if self not in mapping:
            raise ValueError(f"Unsupported dtype {self}")
        return mapping[self]


class LLMEdgeManager:
    """
    Host a torch.nn.Module for LLM model and facilitates exporting to ExecuTorch.
    """

    def __init__(
        self,
        model,
        modelname,
        max_seq_len,
        dtype,
        use_kv_cache,
        example_inputs,
        enable_dynamic_shape: bool = False,
        verbose: bool = False,
        metadata: Optional[dict] = None,
        dynamic_shapes: Optional[Any] = None,
    ):
        self.model = model
        # graph module returned from capture_pre_autograd_graph
        self.pre_autograd_graph_module: Optional[torch.fx.GraphModule] = None
        self.modelname = modelname
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.example_inputs = example_inputs
        self.use_kv_cache = use_kv_cache
        self.enable_dynamic_shape = enable_dynamic_shape
        self.verbose = verbose
        self.metadata = metadata
        self.applied_source_transforms = []
        self.edge_manager: Optional[EdgeProgramManager] = None
        self.export_program = None
        self.output_dir = "."
        self.dynamic_shapes = dynamic_shapes
        self._saved_pte_filename = None

    def set_output_dir(self, output_dir: str) -> "LLMEdgeManager":
        """
        Set the directory where the .pte file will be saved.
        Args:
            output_dir (str): The directory to store the .pte file.
        """
        self.output_dir = output_dir
        return self

    def to_dtype(self, dtype_override: Optional[DType]) -> "LLMEdgeManager":
        """
        Convert the model to the specified dtype.
        Args:
            dtype_override (Optional[DType]): Override the dtype of the model.
        """
        assert not dtype_override or isinstance(
            dtype_override, DType
        ), "Override dtype needs to be of type <DType>"
        if dtype_override is not None and dtype_override != self.dtype:
            torch_dtype = dtype_override.to_torch_dtype()
            logging.info(f"model.to {torch_dtype}")
            self.model = self.model.to(dtype=torch_dtype)
            self.dtype = dtype_override
        return self

    def source_transform(
        self, transforms: List[Callable[[torch.nn.Module], torch.nn.Module]]
    ) -> "LLMEdgeManager":
        """
        Apply source transforms to the model. The transforms are callables that
        takes nn.Module as input and returns nn.Module.
        Args:
            transforms (List[Callable[[torch.nn.Module], torch.nn.Module]]): A
                list of source transforms.
        """
        for transform in transforms:
            self.model = transform(self.model)
        self.applied_source_transforms.extend(transforms)

        if self.verbose:
            logging.info(f"Applied source transforms: {self.applied_source_transforms}")
        return self

    def _get_dynamic_shape(self) -> Any:
        if self.dynamic_shapes:
            return self.dynamic_shapes

        dim = torch.export.Dim("token_dim", max=self.max_seq_len - 1)

        if not self.use_kv_cache:
            # Only one input argument: tokens
            self.dynamic_shapes = ({1: dim},)
        elif self.enable_dynamic_shape:
            # Two input arguments: tokens and input_pos but input_pos is static shape
            self.dynamic_shapes = ({1: dim}, {0: 1})
        else:
            # Two input arguments: tokens and input_pos but both are of static shape
            self.dynamic_shapes = None
        return self.dynamic_shapes

    def _get_edge_config(self) -> EdgeCompileConfig:
        edge_config = EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_type_promotion=bool(self.dtype == DType.fp16),
            _skip_dim_order=True,
        )
        return edge_config

    def capture_pre_autograd_graph(self) -> "LLMEdgeManager":
        dynamic_shape = self._get_dynamic_shape()
        # 1. torch.nn.attention.sdpa_kernel([SDPBackend.MATH]) is for bypassing the dynamo error when tracing
        # 2. torch.no_grad() is for getting rid of the dropout (not sure why training ops will show up)
        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
            self.pre_autograd_graph_module = capture_pre_autograd_graph(
                self.model, self.example_inputs, dynamic_shapes=dynamic_shape
            )
        return self

    def pt2e_quantize(self, quantizers: Optional[List[Quantizer]]) -> "LLMEdgeManager":
        """
        Quantize the model via pt2e flow and retrieve LLMEdgeManager including the quantized model.
        Args:
            quantizers (Optional[List[Quantizer]]): A list of quantizers.
        """
        assert (
            self.edge_manager is None
        ), "export_to_edge is already called, please call pt2e_quantize before export_to_edge"
        logging.info(f"Using pt2e {quantizers} to quantizing the model...")

        # 1. torch.nn.attention.sdpa_kernel([SDPBackend.MATH]) is for bypassing the dynamo error when tracing
        # 2. torch.no_grad() is for getting rid of the dropout (not sure why training ops will show up)
        if quantizers:
            with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
                if self.verbose:
                    logging.info(f"Applied quantizers: {quantizers}")
                composed_quantizer = ComposableQuantizer(quantizers)
                assert (
                    self.pre_autograd_graph_module is not None
                ), "Please run capture_pre_autograd_graph first"
                m = prepare_pt2e(self.pre_autograd_graph_module, composed_quantizer)
                # Calibrate
                m(*self.example_inputs)
                m = convert_pt2e(m)
                DuplicateDynamicQuantChainPass()(m)
                self.pre_autograd_graph_module = m
            return self
        else:
            logging.info("No quantizer provided, passing...")
            return self

    def export_to_edge(self) -> "LLMEdgeManager":
        """
        Export the model to Edge dialect and retrieve a LLMEdgeManager.
        """
        dynamic_shape = self._get_dynamic_shape()
        edge_config = self._get_edge_config()

        # 1. torch.nn.attention.sdpa_kernel([SDPBackend.MATH]) is for bypassing the dynamo error when tracing
        # 2. torch.no_grad() is for getting rid of the dropout (not sure why training ops will show up)
        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
            if self.pre_autograd_graph_module is None:
                self.pre_autograd_graph_module = capture_pre_autograd_graph(
                    self.model, self.example_inputs, dynamic_shapes=dynamic_shape
                )
            self.edge_manager = export_to_edge(
                self.pre_autograd_graph_module,
                self.example_inputs,
                dynamic_shapes=dynamic_shape,
                edge_constant_methods=self.metadata,
                edge_compile_config=edge_config,
                verbose=self.verbose,
            )
        return self

    def to_backend(self, partitioners: Optional[List[Partitioner]]) -> "LLMEdgeManager":
        """
        Partition the model and lower to different backends. The signature is
        aligned with the signature of `to_backend` method of EdgeManager.
        Args:
            partitioner (Optional[Partitioner]): One or more
                partitioner to be sent to EdgeManager.to_backend().
        """
        if partitioners is None:
            logging.info("No partitioner provided, passing...")
        else:
            for partitioner in partitioners:
                if partitioner is not None:
                    assert (
                        self.edge_manager is not None
                    ), "Need to run export_to_edge() first"
                    self.edge_manager = self.edge_manager.to_backend(partitioner)
                    if self.verbose:
                        logging.info(
                            format_delegated_graph(
                                self.edge_manager.exported_program().graph_module
                            )
                        )
                        logging.info(f"Applied partitioners: {partitioner}")
                else:
                    logging.info("No partitioner provided, passing...")
                    continue

        return self

    def to_executorch(self) -> "LLMEdgeManager":
        """
        Lower the model to executorch and get an ExecutorchProgram.
        """
        assert self.edge_manager, "Need to run export_to_edge() first"
        self.export_program = self.edge_manager.to_executorch(
            ExecutorchBackendConfig(
                extract_constant_segment=True,
                extract_delegate_segments=True,
                passes=[
                    QuantFusionPass(),
                ],
                memory_planning_pass=MemoryPlanningPass(
                    "greedy", alloc_graph_input=False
                ),
                sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
            )
        )
        logging.info(
            "Required memory for activation in bytes: {}".format(
                self.export_program._emitter_output.program.execution_plan[
                    0
                ].non_const_buffer_sizes
            ),
        )
        return self

    def save_to_pte(self, output_name: str) -> None:
        """
        Save the model to a .pte file.
        Args:
            output_name (Optional[str]): The name of the .pte file.
        """
        assert output_name, "Need a valid output name"
        filename = save_pte_program(self.export_program, output_name, self.output_dir)
        self._saved_pte_filename = filename

    def get_saved_pte_filename(self) -> Optional[str]:
        """
        Return the filename of the most recenet saved .pte file. Return None if the model is not saved.
        """
        return self._saved_pte_filename
