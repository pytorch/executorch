# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Providing builders for LLM models. These builders help user to build LLM
# eager models, apply source transformations and quantization and export them to
# ExecuTorch.

# pyre-unsafe

import contextlib
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import patch

import torch
from executorch.backends.transforms.duplicate_dynamic_quant_chain import (
    DuplicateDynamicQuantChainPass,
)
from executorch.backends.xnnpack._passes.convert_to_linear import ConvertToLinearPass
from executorch.exir import EdgeProgramManager
from executorch.exir.backend.partitioner import Partitioner

from executorch.exir.backend.utils import format_delegated_graph
from executorch.exir.capture._config import EdgeCompileConfig, ExecutorchBackendConfig

from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass

from executorch.extension.export_util.utils import export_to_edge, save_pte_program
from executorch.extension.llm.tokenizer.utils import get_tokenizer
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer import Quantizer
from torch.ao.quantization.quantizer.composable_quantizer import ComposableQuantizer
from torch.export import export_for_training
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
            DType.bf16: torch.bfloat16,
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
        example_kwarg_inputs: Optional[Dict] = None,
        args: Optional[Any] = None,
        enable_dynamic_shape: bool = False,
        generate_full_logits: bool = False,
        calibration_tasks: Optional[List[str]] = None,
        calibration_limit: Optional[int] = None,
        calibration_seq_length: Optional[int] = None,
        calibration_data: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        verbose: bool = False,
        metadata: Optional[dict] = None,
        dynamic_shapes: Optional[Any] = None,
    ):
        self.model = model
        # graph module returned from export()
        self.pre_autograd_graph_module: Optional[torch.fx.GraphModule] = None
        self.modelname = modelname
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.example_inputs = example_inputs
        self.example_kwarg_inputs = example_kwarg_inputs
        self.use_kv_cache = use_kv_cache
        self.generate_full_logits = generate_full_logits
        self.enable_dynamic_shape = enable_dynamic_shape
        self.verbose = verbose
        self.metadata = metadata
        self.applied_source_transforms = []
        self.edge_manager: Optional[EdgeProgramManager] = None
        self.export_program = None
        self.output_dir = "."
        self.dynamic_shapes = dynamic_shapes
        self._saved_pte_filename = None
        self.args = args
        self.calibration_tasks = calibration_tasks
        self.calibration_limit = calibration_limit
        self.calibration_seq_length = calibration_seq_length
        self.calibration_data = calibration_data
        self.tokenizer_path = tokenizer_path

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
        logging.info(f"Model after source transforms: {self.model}")
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

    def export(self) -> "LLMEdgeManager":
        dynamic_shape = self._get_dynamic_shape()
        # 1. torch.nn.attention.sdpa_kernel([SDPBackend.MATH]) is for bypassing the dynamo error when tracing
        # 2. torch.no_grad() is for getting rid of the dropout (not sure why training ops will show up)
        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
            if hasattr(self.args, "qnn") and self.args.qnn:
                # TODO: this is temporary, as qnn flow does not work with new, non-functional export IR.
                # See issue: https://github.com/pytorch/executorch/issues/7373

                with patch.object(
                    torch._utils_internal,
                    "export_training_ir_rollout_check",
                    return_value=False,
                ):
                    # TODO: this is temporary and export_for_training doesn't work with qnn either. We need a
                    # functional graph. See issue https://github.com/pytorch/executorch/pull/4627 for more details
                    exported_module = torch.export.export(
                        self.model,
                        self.example_inputs,
                        self.example_kwarg_inputs,
                        dynamic_shapes=dynamic_shape,
                        strict=True,
                    )
            else:
                logging.info("Exporting with:")
                logging.info(f"inputs: {self.example_inputs}")
                logging.info(f"kwargs: {self.example_kwarg_inputs}")
                logging.info(f"dynamic shapes: {dynamic_shape}")
                exported_module = export_for_training(
                    self.model,
                    self.example_inputs,
                    kwargs=self.example_kwarg_inputs,
                    dynamic_shapes=dynamic_shape,
                )
            # pyre-fixme[8]: Attribute has type `Optional[GraphModule]`; used as
            #  `Module`.
            self.pre_autograd_graph_module = exported_module.module()
            if hasattr(self.args, "export_only") and self.args.export_only:
                torch.export.save(exported_module, self.args.output_name)

        return self

    def pt2e_calibrate(
        self,
        prepared_module,
        calibration_tasks,
        calibration_limit,
        calibration_seq_length,
        calibration_data,
        tokenizer_path,
    ):
        logging.info("Run calibration...")
        try:
            from executorch.examples.models.llama.eval_llama_lib import (
                GraphModuleEvalWrapper,
            )
            from lm_eval.evaluator import simple_evaluate
        except ImportError:
            raise ImportError(
                "Please install the llm eval dependency via examples/models/llama/install_requirements.sh"
            )

        tokenizer = get_tokenizer(tokenizer_path)

        def calibrate_template(
            module: torch.fx.GraphModule, tokenizer, prompts: str, max_len: int
        ):
            # TODO: change criteria & support batch inputs if necessary
            pos = torch.tensor(0, dtype=torch.int64)
            token_list = tokenizer.encode(prompts, bos=True, eos=False)

            with torch.no_grad():
                while token_list[-1] != tokenizer.eos_id and pos < max_len:
                    logits = module(
                        torch.full((1, 1), token_list[pos]),
                        torch.tensor((pos,)),
                    )
                    pos += 1
                    if pos >= len(token_list):
                        if self.generate_full_logits:
                            token_list.append(
                                torch.argmax(logits[:, -1], dim=-1).item()
                            )
                        else:
                            token_list.append(torch.argmax(logits[:], dim=-1).item())

        calibrate_template(
            module=prepared_module,
            tokenizer=tokenizer,
            prompts=calibration_data,
            max_len=calibration_seq_length,
        )

        eval_wrapper = GraphModuleEvalWrapper(
            model=prepared_module,
            tokenizer=tokenizer,
            max_seq_length=calibration_seq_length,
            use_kv_cache=self.use_kv_cache,
            generate_full_logits=self.generate_full_logits,
            enable_dynamic_shape=self.enable_dynamic_shape,
        )

        # Evaluate the model
        with torch.no_grad():
            eval_results = simple_evaluate(
                model=eval_wrapper,
                tasks=calibration_tasks,
                limit=calibration_limit,
            )

        for task, res in eval_results["results"].items():
            print(f"{task}: {res}")
        logging.info("Calibration finish...")

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
                ), "Please run export() first"
                m = prepare_pt2e(self.pre_autograd_graph_module, composed_quantizer)
                logging.info(
                    f"Calibrating with tasks: {self.calibration_tasks}, limit: {self.calibration_limit}, calibration_data: {self.calibration_data}, tokenizer_path: {self.tokenizer_path}, seq_length: {self.calibration_seq_length}"
                )
                # Calibrate
                if (
                    self.calibration_tasks is not None
                    and self.calibration_limit is not None
                    and self.calibration_seq_length is not None
                    and self.calibration_data is not None
                    and self.tokenizer_path is not None
                ):
                    logging.info(
                        f"Calibrating with tasks: {self.calibration_tasks}, limit: {self.calibration_limit}, calibration_data: {self.calibration_data}, tokenizer_path: {self.tokenizer_path}, seq_length: {self.calibration_seq_length}"
                    )
                    self.pt2e_calibrate(
                        prepared_module=m,
                        calibration_tasks=self.calibration_tasks,
                        calibration_limit=self.calibration_limit,
                        calibration_seq_length=self.calibration_seq_length,
                        calibration_data=self.calibration_data,
                        tokenizer_path=self.tokenizer_path,
                    )
                else:
                    logging.info(
                        "No calibration provided, using dummy input to calibrate..."
                    )
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
                # Run export() if it didn't run
                self.export()

            override_export_behaviour = contextlib.nullcontext()
            if hasattr(self.args, "qnn") and self.args.qnn:
                override_export_behaviour = patch.object(
                    torch._utils_internal,
                    "export_training_ir_rollout_check",
                    return_value=False,
                )

            with override_export_behaviour:
                self.edge_manager = export_to_edge(
                    self.pre_autograd_graph_module,  # pyre-fixme[6]
                    self.example_inputs,
                    example_kwarg_inputs=self.example_kwarg_inputs,
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
            partitioners (Optional[List[Partitioner]]): One or more
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
                extract_delegate_segments=True,
                passes=[
                    # If there are Linear operations left in the graph, let's execute
                    # them with the optimized op_linear rather than materializing a
                    # transpose followed by a regular op_mm.
                    ConvertToLinearPass(),
                    QuantFusionPass(),
                ],
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
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
