# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Providing builders for Llama2 models. These builders help user to build Llama2
# eager models, apply source transformations and quantization and export them to
# ExecuTorch.

import json
import logging
from enum import Enum
from json import JSONDecodeError
from typing import Any, Callable, List, Optional

import torch
from executorch.backends.transforms.duplicate_dynamic_quant_chain import (
    DuplicateDynamicQuantChainPass,
)
from executorch.exir import EdgeProgramManager
from executorch.exir.backend.partitioner import Partitioner

from executorch.exir.backend.utils import print_delegated_graph
from executorch.exir.capture._config import EdgeCompileConfig, ExecutorchBackendConfig

from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer import Quantizer
from torch.ao.quantization.quantizer.composable_quantizer import ComposableQuantizer
from torch.nn.attention import SDPBackend

from ...portable.utils import export_to_edge, save_pte_program
from ..model_factory import EagerModelFactory

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


class WeightType(Enum):
    LLAMA = "LLAMA"
    FAIRSEQ2 = "FAIRSEQ2"


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


def load_llama_model(
    *,
    checkpoint: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    params_path: str,
    use_kv_cache: bool = False,
    use_sdpa_with_kv_cache: bool = False,
    weight_type: WeightType = WeightType.LLAMA,
    verbose: bool = False,
    max_seq_len: int = 128,
) -> "LlamaEdgeManager":
    """
    A helper util that builds a Llama2 model. It returns a LlamaEdgeManager that
    can help further lower the model to ExecuTorch.
    Returns:
        An instance of LlamaEdgeManager which contains the eager mode model.
    """
    assert (
        checkpoint or checkpoint_dir
    ) and params_path, "Both checkpoint/checkpoint_dir and params can't be empty"
    logging.info(
        f"Loading model with checkpoint={checkpoint}, params={params_path}, use_kv_cache={use_kv_cache}, weight_type={weight_type}"
    )
    model, example_inputs, _ = EagerModelFactory.create_model(
        "llama2",
        "Llama2Model",
        checkpoint=checkpoint,
        checkpoint_dir=checkpoint_dir,
        params=params_path,
        use_kv_cache=use_kv_cache,
        use_sdpa_with_kv_cache=use_sdpa_with_kv_cache,
        fairseq2=weight_type == WeightType.FAIRSEQ2,
        max_seq_len=max_seq_len,
    )
    state_dict = model.state_dict()
    dtype = state_dict[next(iter(state_dict))].dtype
    assert dtype in [
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ], f"Only support bfloat16, fp16 or fp32 got {dtype}"
    logging.info(f"Loaded model with dtype={dtype}")

    if dtype == torch.bfloat16:
        dtype = DType.bf16
    elif dtype == torch.float16:
        dtype = DType.fp16
    elif dtype == torch.float32:
        dtype = DType.fp32
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    return LlamaEdgeManager(
        model=model,
        weight_type=weight_type,
        dtype=dtype,
        use_kv_cache=use_kv_cache,
        use_sdpa_with_kv_cache=use_sdpa_with_kv_cache,
        example_inputs=example_inputs,
        verbose=verbose,
    )


class LlamaEdgeManager:
    """
    Host a torch.nn.Module for Llama model and facilitates exporting to ExecuTorch.
    """

    def __init__(
        self,
        model,
        weight_type,
        dtype,
        use_kv_cache,
        use_sdpa_with_kv_cache,
        example_inputs,
        verbose: bool = False,
    ):
        self.model = model
        self.weight_type = weight_type
        self.dtype = dtype
        self.example_inputs = example_inputs
        self.use_kv_cache = use_kv_cache
        self.use_sdpa_with_kv_cache = use_sdpa_with_kv_cache
        self.metadata = None
        self.verbose = verbose
        self.applied_source_transforms = []
        self.edge_manager: Optional[EdgeProgramManager] = None
        self.export_program = None
        self.output_dir = "."

    def set_metadata(self, metadata: Optional[dict]) -> "LlamaEdgeManager":
        """
        Set the metadata that will be serialized into .pte file.
        Args:
            metadata (Optional[dict]): Metadata for the model.
        """
        self.metadata = metadata
        return self

    def set_output_dir(self, output_dir: str) -> "LlamaEdgeManager":
        """
        Set the directory where the .pte file will be saved.
        Args:
            output_dir (str): The directory to store the .pte file.
        """
        self.output_dir = output_dir
        return self

    def to_dtype(self, dtype_override: Optional[DType]) -> "LlamaEdgeManager":
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
    ) -> "LlamaEdgeManager":
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
        dim = torch.export.Dim("token_dim", max=self.model.params.max_seq_len - 1)
        if self.use_kv_cache:
            return None
        else:
            return ({1: dim},)

    def _get_edge_config(self) -> EdgeCompileConfig:
        edge_config = EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_type_promotion=bool(self.dtype == DType.fp16),
        )
        return edge_config

    def _get_metadata(self):
        params = self.model.params
        is_fairseq2 = self.weight_type == WeightType.FAIRSEQ2
        metadata = {
            "append_eos_to_prompt": is_fairseq2,  # For language llama, tell the runtime to always append EOS token(s) to prompt.
            "get_bos_id": 3 if is_fairseq2 else 1,
            "get_dtype": 5 if self.dtype == DType.fp16 else 6,
            "get_eos_id": 3 if is_fairseq2 else 2,
            "get_head_dim": params.dim // params.n_heads,
            "get_max_batch_size": params.max_batch_size,
            "get_max_seq_len": params.max_seq_len,
            "get_n_bos": 1,
            "get_n_eos": 2 if is_fairseq2 else 1,
            "get_n_kv_heads": params.n_kv_heads,
            "get_n_layers": params.n_layers,
            "get_vocab_size": params.vocab_size,
            "use_kv_cache": self.use_kv_cache,
            "use_sdpa_with_kv_cache": self.use_sdpa_with_kv_cache,
        }
        if self.metadata:
            try:
                extra = json.loads(self.metadata)
                for k, v in extra.items():
                    metadata[k] = v
            except JSONDecodeError:
                logging.error("Invalid metadata, should be a valid JSON string")
        self.metadata = metadata
        return self.metadata

    def export_to_edge(
        self, quantizers: Optional[List[Quantizer]]
    ) -> "LlamaEdgeManager":
        """
        Export the model to Edge dialect and retrieve a EdgeManager.
        Args:
            quantizers (Optional[List[Quantizer]]): A list of quantizers.
        """
        dynamic_shape = self._get_dynamic_shape()
        edge_config = self._get_edge_config()
        metadata = self._get_metadata()

        # 1. torch.nn.attention.sdpa_kernel([SDPBackend.MATH]) is for bypassing the dynamo error when tracing
        # 2. torch.no_grad() is for getting rid of the dropout (not sure why training ops will show up)
        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
            m = capture_pre_autograd_graph(
                self.model, self.example_inputs, dynamic_shapes=dynamic_shape
            )
            if quantizers:
                if self.verbose:
                    logging.info(f"Applied quantizers: {quantizers}")
                composed_quantizer = ComposableQuantizer(quantizers)
                m = prepare_pt2e(m, composed_quantizer)
                # Calibrate
                m(*self.example_inputs)
                m = convert_pt2e(m)
                DuplicateDynamicQuantChainPass()(m)
            self.edge_manager = export_to_edge(
                m,
                self.example_inputs,
                dynamic_shapes=dynamic_shape,
                edge_constant_methods=metadata,
                edge_compile_config=edge_config,
                verbose=True,
            )
        return self

    def to_backend(
        self, partitioners: Optional[List[Partitioner]]
    ) -> "LlamaEdgeManager":
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
                            print_delegated_graph(
                                self.edge_manager.exported_program().graph_module
                            )
                        )
                        logging.info(f"Applied partitioners: {partitioner}")
                else:
                    logging.info("No partitioner provided, passing...")
                    continue

        return self

    def to_executorch(self) -> "LlamaEdgeManager":
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
        save_pte_program(self.export_program, output_name, self.output_dir)
