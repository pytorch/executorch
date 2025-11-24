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
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from executorch.backends.transforms.duplicate_dynamic_quant_chain import (
    DuplicateDynamicQuantChainPass,
)
from executorch.backends.xnnpack._passes.convert_to_linear import ConvertToLinearPass
from executorch.exir import EdgeProgramManager, to_edge_transform_and_lower
from executorch.exir.backend.partitioner import Partitioner

from executorch.exir.backend.utils import format_delegated_graph
from executorch.exir.capture._config import EdgeCompileConfig, ExecutorchBackendConfig

from executorch.exir.pass_base import ExportPass
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass

from executorch.extension.export_util.utils import export_to_edge, save_pte_program

from executorch.extension.llm.export.export_passes import RemoveRedundantTransposes
from pytorch_tokenizers import get_tokenizer
from torch.export import export, ExportedProgram
from torch.nn.attention import SDPBackend
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer import ComposableQuantizer, Quantizer
from torchao.utils import unwrap_tensor_subclass

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

    @staticmethod
    def from_torch_dtype(dtype: torch.dtype):
        mapping = {
            torch.float32: DType.fp32,
            torch.float16: DType.fp16,
            torch.bfloat16: DType.bf16,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported torch.dtype {dtype}")
        return mapping[dtype]


class LLMEdgeManager:
    """
    Host a torch.nn.Module for LLM model and facilitates exporting to ExecuTorch.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        modelname: str,
        max_seq_len: int,
        use_kv_cache: bool,
        example_inputs: Tuple[torch.Tensor, ...],
        dtype: Optional[DType] = None,
        example_kwarg_inputs: Optional[Dict] = None,
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
        save_exported_program: bool = False,
        generate_etrecord: bool = False,
    ):
        # Store necessary constructor arguments.
        self.model = model
        self.modelname = modelname
        self.max_seq_len = max_seq_len
        self.use_kv_cache = use_kv_cache
        self.example_inputs = example_inputs
        self.dtype = dtype
        self.example_kwarg_inputs = example_kwarg_inputs
        self.enable_dynamic_shape = enable_dynamic_shape
        self.generate_full_logits = generate_full_logits
        self.calibration_tasks = calibration_tasks
        self.calibration_limit = calibration_limit
        self.calibration_seq_length = calibration_seq_length
        self.calibration_data = calibration_data
        self.tokenizer_path = tokenizer_path
        self.verbose = verbose
        self.metadata = metadata if metadata is not None else {}
        self.metadata["get_max_seq_len"] = max_seq_len
        self.dynamic_shapes = dynamic_shapes
        self.save_exported_program = save_exported_program
        self.generate_etrecord = generate_etrecord

        # Note: treat this as the source of truth for the result of
        # torch.export'ing a model. If the overall ExportedProgram is needed,
        # make sure to re-export this graph module to persist any changes. See
        # https://github.com/pytorch/pytorch/blob/main/torch/export/exported_program.py#L921
        self.pre_autograd_graph_module: Optional[torch.nn.Module] = None
        self.edge_manager: Optional[EdgeProgramManager] = None
        self.canonical_passes = [
            RemoveRedundantTransposes()
        ]  # Graph transformations optimizations.
        self.export_program = None  # Final result of lowering to executorch.
        self.output_dir = "."
        self._saved_pte_filename = None

        # Try to resolve dynamic shapes if not specified explicitly.
        if not self.dynamic_shapes and self.enable_dynamic_shape:
            if not self.use_kv_cache:
                # Only one input argument: tokens
                # Here we -1 due to export limitation: https://gist.github.com/larryliu0820/419022a57e24d5e64150e325a685eaad
                self.dynamic_shapes = (
                    {1: torch.export.Dim("token_dim", max=self.max_seq_len - 1)},
                )
            else:
                # Two input arguments: tokens and input_pos but input_pos is static shape.

                self.dynamic_shapes = (
                    {1: torch.export.Dim("token_dim", max=self.max_seq_len)},
                    {"input_pos": {0: 1}},
                )

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

        if self.verbose:
            logging.info(f"Applied source transforms: {transforms}")
        logging.info(f"Model after source transforms: {self.model}")
        return self

    def _get_dynamic_shape(self) -> Any:
        return self.dynamic_shapes

    def _get_edge_config(self) -> EdgeCompileConfig:
        edge_config = EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        )
        return edge_config

    def _export(self, module: Optional[torch.nn.Module] = None) -> ExportedProgram:
        if module is not None:
            unwrap_tensor_subclass(module)
        else:
            unwrap_tensor_subclass(self.model)

        dynamic_shape = self._get_dynamic_shape()
        # 1. torch.nn.attention.sdpa_kernel([SDPBackend.MATH]) is for bypassing the dynamo error when tracing
        # 2. torch.no_grad() is for getting rid of the dropout (not sure why training ops will show up)
        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
            if module:
                logging.info("Re-exporting with:")
            else:
                logging.info("Exporting with:")
            logging.info(f"inputs: {self.example_inputs}")
            logging.info(f"kwargs: {self.example_kwarg_inputs}")
            logging.info(f"dynamic shapes: {dynamic_shape}")
            exported_module = export(
                self.model if not module else module,
                self.example_inputs,
                kwargs=self.example_kwarg_inputs,
                dynamic_shapes=dynamic_shape,
                strict=True,
            )
        return exported_module

    def export(self) -> "LLMEdgeManager":
        """
        Exports the model pre-autograd. This is not a full export, since it uses
        torch.export.export() to keep autograd-safe ops from getting decomposed.
        The full torch.export() if called later on during to_edge() or
        to_edge_transform_and_lower().
        """
        exported_module = self._export()
        # Need to store the graph module to record transformation passes.
        # Persisting those changes back to an ExportedProgram will require
        # an additional export().
        self.pre_autograd_graph_module = exported_module.module()
        if self.save_exported_program:
            export_output = f"{self.modelname}.pt2"
            logging.info(f"Saving torch.export() result to {export_output}")
            torch.export.save(exported_module, export_output)
        return self

    def run_canonical_optimizations(self):
        """
        Run canonical optimizations (at the moment removing redundant permutes) on the model.
        """
        assert self.pre_autograd_graph_module is not None, "Please run export() first"
        for pass_instance in self.canonical_passes:
            logging.info(f"Running canonical pass: {pass_instance.__class__.__name__}")
            res = pass_instance(self.pre_autograd_graph_module)
            assert res.graph_module is not None, "Pass returned None"
            self.pre_autograd_graph_module = res.graph_module

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
                        {"input_pos": torch.tensor((pos,))},
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
                m = prepare_pt2e(
                    self.pre_autograd_graph_module,  # pyre-ignore[6]
                    composed_quantizer,
                )
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
                    if self.example_kwarg_inputs:
                        m(*self.example_inputs, **self.example_kwarg_inputs)
                    else:
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

    def to_edge_transform_and_lower(
        self, partitioners: Optional[List[Partitioner]]
    ) -> "LLMEdgeManager":
        if partitioners is None:
            logging.info("No partitioner provided, skipping backend lowering...")

        # Need to construct ExportedProgram with the new transformed graph module.
        exported_module = self._export(self.pre_autograd_graph_module)

        edge_config = self._get_edge_config()
        self.edge_manager = to_edge_transform_and_lower(
            exported_module,
            partitioner=partitioners,
            compile_config=edge_config,
            constant_methods=self.metadata,
            generate_etrecord=self.generate_etrecord,
        )
        if self.verbose:
            logging.info(f"Exported graph:\n{self.edge_manager.exported_program()}")
        return self

    def to_executorch(
        self,
        passes: Optional[List[ExportPass]] = None,
        external_constants_tag: Optional[
            Callable[[torch.fx.Node], Optional[str]]
        ] = None,
    ) -> "LLMEdgeManager":
        """
        Lower the model to executorch and get an ExecutorchProgram.
        """
        to_executorch_passes = []
        if passes:
            # pyre-fixme[6]: In call `list.extend`, for 1st positional argument,
            # expected `Iterable[Union[ConvertToLinearPass, QuantFusionPass]]` but
            # got `List[ExportPass]
            to_executorch_passes.extend(passes)

        assert self.edge_manager, "Need to run export_to_edge() first"

        # If there are Linear operations left in the graph, let's execute
        # them with the optimized op_linear rather than materializing a
        # transpose followed by a regular op_mm.
        # TODO: ConvertToLinearPass is not a sound pass and must be called before
        # const propagation.  It requires fixing:
        # https://github.com/pytorch/executorch/issues/10499
        self.edge_manager.transform([ConvertToLinearPass()])

        self.export_program = self.edge_manager.to_executorch(
            ExecutorchBackendConfig(
                extract_delegate_segments=True,
                # pyre-fixme[6]: In call `ExecutorchBackendConfig.__init__`, for
                # argument `passes`, expected `List[typing.Callable[[GraphModule],
                # Optional[PassResult]]]` but got `List[Union[ConvertToLinearPass,
                # QuantFusionPass]]`.
                passes=to_executorch_passes,
                do_quant_fusion_and_const_prop=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
                sym_shape_eval_pass=ConstraintBasedSymShapeEvalPass(),
                external_constants=external_constants_tag,
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
