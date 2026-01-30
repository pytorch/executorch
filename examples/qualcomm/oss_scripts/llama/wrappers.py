from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import time
import types

from dataclasses import dataclass
from enum import Enum
from functools import partial, wraps
from typing import Any, Dict, List, Tuple

import torch

from executorch.backends.qualcomm._passes import FoldQDQ, I64toI32, TagQuantIO
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_capture_program_passes,
)
from executorch.backends.qualcomm._passes.utils import (
    get_passes_dependency_for_capture_program,
)
from executorch.backends.qualcomm.builders.utils import is_graph_output
from executorch.backends.qualcomm.quantizer.custom_annotation import (
    annotate_prefill_kv_output,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.utils.constants import (
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
)
from executorch.backends.qualcomm.utils.utils import (
    convert_linear_to_conv2d,
    get_sdk_build_id,
    is_qnn_sdk_version_less_than,
    to_edge_transform_and_lower_to_qnn,
    update_spill_fill_size,
)
from executorch.devtools.backend_debug import print_delegation_info
from executorch.examples.models.llama.hf_download import (
    download_and_convert_hf_checkpoint,
)
from executorch.examples.models.llama.source_transformation.quantize import (
    get_quant_embedding_transform,
)
from executorch.examples.qualcomm.oss_scripts.llama import (
    LLM_VARIANT_ARCHS,
    LLMModelConfig,
)
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    AUDIO_ENCODER,
    DECODER_GRAPH_NAMES,
    TEXT_DECODER,
    TEXT_EMBEDDING,
    TEXT_EMBEDDING_GRAPH_NAMES,
    TEXT_ENCODER,
    VISION_ENCODER,
)
from executorch.examples.qualcomm.oss_scripts.llama.decoder_utils import (
    graph_module_inference,
)
from executorch.examples.qualcomm.oss_scripts.llama.encoder.encoder_quant_recipe import (
    EncoderQuantRecipe,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.embedding import TextEmbedding
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    LlamaModel,
    ModelArgs,
)
from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import (
    StaticLLMQuantRecipe,
)
from executorch.examples.qualcomm.utils import make_quantizer
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.extension.llm.custom_ops import model_sharding
from executorch.extension.llm.export.builder import DType
from torchao.prototype.spinquant import apply_spinquant
from torchao.quantization.pt2e import MinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from transformers import AutoConfig, AutoModel


def is_node_src_start_with_name(node: torch.fx.Node, prefix: str) -> bool:
    """
    Return True if any NodeSource in node.meta['from_node']
    has a `name` starting with `prefix`.
    """

    def has_source_name_prefix(
        node_src: torch.fx.traceback.NodeSource, prefix: str
    ) -> bool:

        name = getattr(node_src, "name", None)
        if isinstance(name, str) and name.startswith(prefix):
            return True

        children = getattr(node_src, "from_node", None)
        if not children:
            return False

        for src in children:
            if has_source_name_prefix(src, prefix):
                return True

        return False

    node_srcs = node.meta.get("from_node", None)
    if not node_srcs:
        return False

    return any(has_source_name_prefix(node_src, prefix) for node_src in node_srcs)


def log_info(func):
    class TimeIt:
        def __init__(self, event):
            self.event = event

        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, type, value, traceback):
            self.time = time.time() - self.start
            logging.info(f"{self.event}{self.time}s")

    @wraps(func)
    def wrapper(cls, *args, **kwargs):
        func_name = f"{cls.__class__.__name__}::{func.__name__}"
        logging.info(f"calling {func_name}")
        with TimeIt(f"{func_name} completed in "):
            func(cls, *args, **kwargs)

    return wrapper


def next_power_of_two(n):
    return 1 if n == 0 else 2 ** math.ceil(math.log2(n))


class Processor:
    _next_handler = None

    def set_next(self, processor) -> Processor:
        self._next_handler = processor
        return processor

    def process(self, request: Any):
        if self._next_handler:
            return self._next_handler.process(request)


@dataclass
class Request:
    @dataclass
    class CalibrationData:
        datasets: List[Tuple[torch.Tensor]] = None
        intermediate_outputs: List[Tuple[torch.Tensor]] = None
        qdq_intermediate_outputs: List[Tuple[torch.Tensor]] = None

    @dataclass
    class Data:
        compile_spec: List[CompileSpec] = None
        pte_filename: str = None
        custom_annotation: Any = ()
        calibration_data: Request.CalibrationData = None
        tokenizer: callable = None

    method_name: str
    method_data: Dict[str, Data]


class Component(Processor):
    def process(self, request: Request) -> Request:
        getattr(self, request.method_name)(request)
        super().process(request)

    def compile(self, request: Request):
        return

    def quantize(self, request: Request):
        return


class TextDecoder(Component):
    class Mode(Enum):
        PREFILL = 1
        DECODE = 2

    def __init__(
        self,
        control_args: argparse.Namespace,
        config: LLMModelConfig,
        mode: Mode,
        apply_embedding: bool = False,
    ):
        self.control_args = control_args
        self.config = config
        self.mode = mode
        self.passes_job = get_capture_program_passes()
        self.dep_table = get_passes_dependency_for_capture_program()
        self.meta = {}
        self.quant_recipe: StaticLLMQuantRecipe = (
            self.config.quant_recipe(True) if self.config.quant_recipe else None
        )

        # For multimodal embedding
        self._modality_placeholder_token_id = None
        self.apply_embedding = apply_embedding
        self.tok_embedding_passes_job = (
            get_capture_program_passes() if apply_embedding else None
        )
        self.tok_embedding_dep_table = (
            get_passes_dependency_for_capture_program() if apply_embedding else None
        )

        # load static llama model args
        params_path = (
            config.params_path if control_args.params is None else control_args.params
        )
        with open(params_path) as f:
            self.model_args = self._process_model_args(ModelArgs(**json.load(f)))

        # prepare instance
        self.tok_embedding = None
        self.decoder = None
        if (instance := self._prepare_model()) is not None:
            self.tok_embedding, self.decoder = instance
            self.meta = self.decoder.get_metadata()

        # check if sharding required
        if self.decoder and self.config.num_sharding > 1:
            SplitGraph, setting = model_sharding.get_split_graph_pass(
                self.meta["get_n_layers"],
                shares=self.config.num_sharding,
            )
            self.passes_job[SplitGraph] = setting
            self.dep_table[SplitGraph] = [FoldQDQ]
            self.dep_table[TagQuantIO] = [SplitGraph]

    def _process_model_args(self, model_args: ModelArgs):
        # TODO: support batch inputs if necessary
        model_args.max_batch_size = 1
        model_args.max_seq_len = self.control_args.max_seq_len
        model_args.use_kv_cache = (
            self.control_args.max_seq_len != self.control_args.prefill_ar_len
        )
        model_args.enable_r3 = self.config.r3
        model_args.kv_io_bit_width = self.quant_recipe.get_kv_io_bit_width()
        if self.config.masked_softmax:
            if is_qnn_sdk_version_less_than("2.35"):
                logging.warning(
                    f"Masked softmax is supported after QNN SDK 2.35. Given sdk version {get_sdk_build_id()}"
                    " is lower the target version. Disabling the feature."
                )
                model_args.enable_masked_softmax = False
            else:
                model_args.enable_masked_softmax = True

        return model_args

    def _prepare_model(self):  # noqa: C901
        if (instance := self._get_model_instance()) is None:
            return None
        tok_embedding, decoder = instance
        # load parameters for HF models
        if self.control_args.checkpoint is None:
            checkpoint = download_and_convert_hf_checkpoint(
                self.config.repo_id,
                self.config.convert_weights.__func__,
            )
            state_dict = torch.load(
                checkpoint, weights_only=True, map_location="cpu", mmap=True
            )
            if self.control_args.decoder_model in {
                "gemma-2b",
                "gemma2-2b",
                "gemma3-1b",
            }:
                for k, v in state_dict.items():
                    if "norm" not in k:
                        continue
                    # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
                    # See https://github.com/huggingface/transformers/pull/29402
                    state_dict[k] = v.float() + torch.ones(v.shape, dtype=torch.float32)
        else:
            state_dict = torch.load(
                self.control_args.checkpoint,
                weights_only=True,
                map_location="cpu",
                mmap=True,
            )
            if "model" in state_dict:
                state_dict = state_dict["model"]

            if self.control_args.decoder_model == "stories260k":
                state_dict = {
                    k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
                }

        # change to HF weight to improve the performance of RoPE in HTP backend.
        if self.config.transform_weight:

            def permute(w, heads, partial_rotary_dim):
                dim_0 = w.size(0)
                dim_1 = w.size(1)
                transformed_weight = (
                    w.view(
                        heads, -1, dim_0 // heads // 2 // partial_rotary_dim, 2, dim_1
                    )
                    .transpose(2, 3)
                    .reshape(dim_0, dim_1)
                )
                return transformed_weight

            # TODO: handle cases where input size isn't divisible.
            partial_rotary_dim = int(1 // self.model_args.partial_rotary_factor)
            for layer_i in range(decoder.n_layers):
                state_dict[f"layers.{layer_i}.attention.wq.weight"] = permute(
                    state_dict[f"layers.{layer_i}.attention.wq.weight"],
                    decoder.n_heads,
                    partial_rotary_dim,
                )
                state_dict[f"layers.{layer_i}.attention.wk.weight"] = permute(
                    state_dict[f"layers.{layer_i}.attention.wk.weight"],
                    decoder.n_kv_heads,
                    partial_rotary_dim,
                )

        decoder.load_state_dict(state_dict, strict=True, assign=True)

        # apply spin quant if required
        if any([self.config.r1, self.config.r2]):
            decoder.config = types.SimpleNamespace(
                dim=decoder.dim,
                head_dim=decoder.dim // decoder.n_heads,
                n_local_heads=decoder.n_heads,
                intermediate_size=4 * decoder.dim,
            )
            apply_spinquant(
                decoder,
                use_r1=self.config.r1,
                use_r2=self.config.r2,
                use_r4=False,
                pretrained_rotation_path=None,
                qkv_split=True,
            )

        # perform model transformation
        for layer in decoder.layers:
            if getattr(layer.attention, "prepare_attention_conv", None):
                layer.attention.prepare_attention_conv()
            if getattr(layer.feed_forward, "prepare_feedfoward_conv", None):
                layer.feed_forward.prepare_feedfoward_conv()

        decoder = convert_linear_to_conv2d(decoder)

        # check dtype override
        if self.control_args.dtype_override is not None:
            dtype_override = DType[self.control_args.dtype_override]
            decoder = decoder.to(dtype_override.to_torch_dtype())

        # check embedding fallback
        if self.control_args.embedding_quantize:
            decoder = get_quant_embedding_transform(
                embedding_quantize=self.control_args.embedding_quantize
            )(decoder)
            self.passes_job[I64toI32][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
                "skip_node"
            ] = {"tokens"}
            if self.apply_embedding:
                tok_embedding = get_quant_embedding_transform(
                    embedding_quantize=self.control_args.embedding_quantize
                )(tok_embedding)
                self.tok_embedding_passes_job[I64toI32][
                    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY
                ]["skip_node"] = {"tokens"}

        if tok_embedding is not None:
            tok_embedding = tok_embedding.eval()

        return tok_embedding, decoder.eval()

    def _get_model_specific_kwargs(self):
        """
        Retrieve model-specific config required for Static LLaMA.
        This method handles architecture-specific requirements for both Vision-Language Models (VLMs)
        and Language-only Models (LLMs), extracting necessary config from HuggingFace configs.

        """
        kwargs = {}
        # Vision-Language Model (VLM)
        # For multimodal models, we need the special token ID that represents image placeholders
        # in the input sequence. This token is used to mark positions where image embeddings
        # should be inserted during inference.
        if hasattr(self.config, VISION_ENCODER):
            hf_config = AutoConfig.from_pretrained(self.config.repo_id)
            kwargs["modality_placeholder_token_id"] = hf_config.image_token_id
        # TODO: Support Audio modality
        elif hasattr(self.config, AUDIO_ENCODER):
            raise NotImplementedError(
                "Audio encoder modality is not currently supported. "
                "Please provide a valid modality_placeholder_token_id in kwargs."
            )

        return kwargs

    def _get_model_instance(self) -> LlamaModel:
        if self.mode == self.Mode.DECODE:
            ar_len = (
                # To get better performance, we round up to the nearest power of 2.
                next_power_of_two(
                    (self.control_args.window + self.control_args.gcap)
                    * (self.control_args.ngram - 1)
                )
                if self.control_args.model_mode == "lookahead"
                else 1
            )
        else:
            if self.control_args.model_mode == "kv":
                return None
            ar_len = self.control_args.prefill_ar_len
        use_i64_token = self.control_args.embedding_quantize is not None

        # get embedding model
        tok_embedding = None
        if self.apply_embedding:
            auto_model = AutoModel.from_pretrained(
                self.config.repo_id, _attn_implementation="eager"
            )
            tok_embedding = TextEmbedding(
                auto_model.get_input_embeddings().to(torch.float32),
                self.model_args.max_batch_size,
                ar_len,
                self.model_args.vocab_size,
                self.model_args.dim,
                use_i64_token,
            )
        # get decoder model
        self.model_args.max_batch_size = 1
        self.model_args.max_seq_len = self.control_args.max_seq_len
        self.model_args.use_kv_cache = True
        self.model_args.enable_r3 = self.config.r3
        self.model_args.kv_io_bit_width = self.quant_recipe.get_kv_io_bit_width()
        if self.control_args.decoder_model in {"gemma-2b", "gemma3-1b"}:
            # For gemma, we have preprocessed the weight of rmsnorm
            self.model_args.norm_type = "rmsnorm"

        decoder: LlamaModel = LLM_VARIANT_ARCHS.get(
            self.control_args.decoder_model, LlamaModel
        )(
            self.model_args,
            ar_len=ar_len,
            output_new_cache_only=True,
            output_cache=True,
            use_i64_token=use_i64_token,
            **self._get_model_specific_kwargs(),
        )
        # get example input
        self.meta = decoder.get_metadata()
        self.example_input = decoder.get_example_inputs()
        self.get_example_inputs = decoder.get_example_inputs
        self.export_input = (
            self.example_input[0],  # tokens or hidden_states
            *self.example_input[1],  # attn_mask
            *((self.example_input[2],) if decoder.use_kv_cache else []),  # pos_ids
            *(self.example_input[3] if decoder.use_kv_cache else []),  # k_caches
            *(self.example_input[4] if decoder.use_kv_cache else []),  # v_caches
        )
        self.io_shape = {
            # logit output
            (
                decoder.max_batch_size,
                decoder.ar_len,
                decoder.vocab_size,
            ),
        }

        if self.apply_embedding:
            self.tok_embedding_export_input = (
                tok_embedding.get_example_input()
            )  # tokens

        return tok_embedding, decoder

    def _save_logits_quant_attrs(self):
        for node in self.decoder.graph.nodes:
            if node.op == "output":
                for output_node in node.args[0]:
                    if (
                        output_node.target
                        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                    ):
                        source_node = output_node.args[0].args[0]
                        if source_node.meta["val"].size() in self.io_shape:
                            self.meta["get_logits_scale"] = output_node.args[1]
                            self.meta["get_logits_zero_point"] = output_node.args[2]
                            break

    def _save_input_kv_cache_quant_attrs(self):
        input_kv_cache_shape = {
            # single head, k input
            (
                self.meta["get_head_dim"],
                self.meta["get_max_seq_len"] - self.meta["get_ar_len"],
            ),
            # single head, v input
            (
                self.meta["get_max_seq_len"] - self.meta["get_ar_len"],
                self.meta["get_head_dim"],
            ),
        }

        idx = 0
        for node in self.decoder.graph.nodes:
            if (
                node.op == "placeholder"
                and len(users := list(node.users)) == 1
                and "val" in node.meta
            ):
                if node.meta["val"].size()[-2:] in input_kv_cache_shape:
                    scale_cache_name = f"get_k_scale_input_{idx}"
                    zero_point_cache_name = f"get_k_zero_point_input_{idx}"
                    if idx >= self.meta["get_n_layers"]:
                        scale_cache_name = (
                            f"get_v_scale_input_{idx % self.meta['get_n_layers']}"
                        )
                        zero_point_cache_name = (
                            f"get_v_zero_point_input_{idx % self.meta['get_n_layers']}"
                        )
                    self.meta[scale_cache_name] = users[0].args[1]
                    self.meta[zero_point_cache_name] = users[0].args[2]
                    idx += 1

    def _save_output_kv_cache_quant_attrs(self):
        output_kv_cache_shape = {
            (self.meta["get_head_dim"], self.meta["get_ar_len"]),
            (self.meta["get_ar_len"], self.meta["get_head_dim"]),
        }
        k_idx = 0
        v_idx = 0
        for node in self.decoder.graph.nodes:
            if not is_graph_output(node):
                continue
            cache_output_node = node.args[0].args[0]
            if cache_output_node.meta["val"].size()[-2:] in output_kv_cache_shape:
                if is_node_src_start_with_name(cache_output_node, "k_"):
                    self.meta[f"get_k_scale_output_{k_idx}"] = node.args[1]
                    self.meta[f"get_k_zero_point_output_{k_idx}"] = node.args[2]
                    k_idx += 1
                elif is_node_src_start_with_name(cache_output_node, "v_"):
                    self.meta[f"get_v_scale_output_{v_idx}"] = node.args[1]
                    self.meta[f"get_v_zero_point_output_{v_idx}"] = node.args[2]
                    v_idx += 1

    def _tag_ios(self, node, fixed_point_type):
        # shape of k caches and v caches
        kv_cache_shape = {
            # single head, kv input
            (self.meta["get_head_dim"], self.meta["get_max_seq_len"]),
            (self.meta["get_max_seq_len"], self.meta["get_head_dim"]),
            # single head, kv output
            (self.meta["get_head_dim"], self.meta["get_ar_len"]),
            (self.meta["get_ar_len"], self.meta["get_head_dim"]),
        }

        atten_mask_shape = {
            (
                self.meta["get_max_batch_size"],
                self.meta["get_ar_len"],
                self.meta["get_max_seq_len"],
            ),
        }

        freq_shape = {
            (self.meta["get_ar_len"], self.meta["get_head_dim"] // 2),
        }

        freq_op = {
            exir_ops.edge.aten.select.int,
        }
        quant_io_type = None

        if node.op == "placeholder":
            if (
                len(users := list(node.users)) == 1
                and users[0].meta["val"].size()[-2:] in kv_cache_shape
            ):
                quant_io_type = fixed_point_type["kv_type"]
            elif node.meta["val"].size() in self.io_shape:
                quant_io_type = fixed_point_type["io_type"]
            elif node.meta["val"].size() in atten_mask_shape:
                quant_io_type = fixed_point_type["io_type"]

        if is_graph_output(node):
            if node.meta["val"].size()[-2:] in kv_cache_shape:
                quant_io_type = fixed_point_type["kv_type"]
            elif node.meta["val"].size() in self.io_shape:
                quant_io_type = fixed_point_type["io_type"]

        # tag sharding io
        if exir_ops.edge.llama.fallback.default in [
            u.target for u in list(node.users.keys())
        ] + [node.target]:
            quant_io_type = fixed_point_type["io_type"]

        # tag select op as quantized tensors for freq_sin and freq_cos. It is caused by sharding
        if node.target in freq_op and node.meta["val"].size() in freq_shape:
            quant_io_type = fixed_point_type["io_type"]

        return quant_io_type

    def _calibrate(
        self,
        model,
        tokenizer,
        event,
        user_calibration_data,
        tok_embedding=None,
        intermediate_outputs=None,
    ):
        """
        Calibrate the model using either task-based evaluation or prompt-based inference.

        This method performs Post-Training Quantization (PTQ) calibration by running inference
        on the model with either:
        1. Task-based datasets by lm_eval for text-only models in perplexity evaluation
        2. User-provided prompts for both text-only and multimodal models

        Args:
            model: The decoder model to calibrate (GraphModule after prepare_pt2e)
            tokenizer: Tokenizer for encoding text inputs
            event: Event name for logging (e.g., "prepare_pt2e", "convert_pt2e")
            tok_embedding: Optional text embedding module (required only for multimodal models)
            intermediate_outputs: Optional pre-computed embeddings from vision/audio encoder
                                 (required only for multimodal models)
        """
        # Determine if this is a multimodal model
        is_multimodal = tok_embedding is not None

        # Determine if task-based calibration is requested
        has_task_calibration = self.control_args.tasks is not None

        # Task-based calibration: Only for text-only LLMs
        # Multimodal models (VLMs) cannot use task-based evaluation currently.
        if has_task_calibration and not is_multimodal:
            graph_module_inference(
                use_kv_cache=self.meta["get_use_kv_cache"],
                get_example_inputs=self.get_example_inputs,
                module=model,
                tokenizer=tokenizer,
                ar_len=self.meta["get_ar_len"],
                max_seq_len=self.meta["get_max_seq_len"],
                tasks=self.control_args.tasks,
                tasks_limit=self.control_args.limit,
                num_fewshot=self.control_args.num_fewshot,
                use_i64_token=self.control_args.embedding_quantize is not None,
                event_name=f"{event}_tasks",
                seq_mse_candidates=self.config.seq_mse_candidates,
            )

        # prepare lookahead config if applicable
        lookahead_config = (
            (self.control_args.window, self.control_args.ngram, self.control_args.gcap)
            if (
                self.mode == self.Mode.DECODE
                and self.control_args.model_mode == "lookahead"
            )
            else None
        )
        # check user's prompt which helps calibrate special token
        for prompt in user_calibration_data:
            graph_module_inference(
                use_kv_cache=self.meta["get_use_kv_cache"],
                get_example_inputs=self.get_example_inputs,
                hidden_states=intermediate_outputs,  # hidden_states for multimodal
                module=model,
                tok_embedding=tok_embedding,
                modality_placeholder_token_id=self.meta.get(
                    "modality_placeholder_token_id", None
                ),
                tokenizer=tokenizer,
                ar_len=self.meta["get_ar_len"],
                max_seq_len=self.meta["get_max_seq_len"],
                prompt=prompt,
                use_i64_token=self.control_args.embedding_quantize is not None,
                event_name=f"{event}_prompt",
                lookahead_config=lookahead_config,
            )

    @log_info
    def quantize(self, request: Request):  # noqa: C901
        if self.quant_recipe is None:
            return

        if self.decoder is None or (
            self.apply_embedding and self.tok_embedding is None
        ):
            return

        # check bit width graph io
        fixed_point_type = {"kv_type": torch.float32, "io_type": torch.float32}
        if self.quant_recipe.get_kv_io_bit_width() == 8:
            fixed_point_type["kv_type"] = torch.uint8
        elif self.quant_recipe.get_kv_io_bit_width() == 16:
            fixed_point_type["kv_type"] = torch.uint16
        else:
            raise RuntimeError(
                f"unknown kv io bit width {self.quant_recipe.get_kv_io_bit_width()}"
            )

        if self.quant_recipe.get_logits_output_bit_width() == 16:
            fixed_point_type["io_type"] = torch.uint16
        else:
            raise RuntimeError(
                f"unknown logits io bit width {self.quant_recipe.get_logits_output_bit_width()}"
            )

        data = request.method_data[TEXT_DECODER]

        image_embedding = None
        if self.apply_embedding:
            # For demo: get first data now
            image_embedding = request.method_data[
                VISION_ENCODER
            ].calibration_data.intermediate_outputs[0]

        quantizer = make_quantizer()
        for custom_annotation in data.custom_annotation:
            self.quant_recipe.recipe.custom_quant_annotations.append(custom_annotation)
        quantizer.recipe = self.quant_recipe

        text_embedding_quantizer = make_quantizer(
            quant_dtype=QuantDtype.use_16a8w,
            per_channel_conv=True,
            per_channel_linear=True,
            act_observer=MinMaxObserver,
        )

        with torch.no_grad():
            # prepare tok embedding model for ptq
            if self.apply_embedding:
                self.tok_embedding = torch.export.export(
                    self.tok_embedding,
                    self.tok_embedding.get_example_input(),
                    strict=True,
                ).module()

            # prepare decoder model for ptq
            self.decoder = torch.export.export(
                self.decoder, self.export_input, strict=True
            ).module()
            self.decoder = prepare_pt2e(self.decoder, quantizer)
            if self.apply_embedding:
                self.tok_embedding = prepare_pt2e(
                    self.tok_embedding, text_embedding_quantizer
                )

            # start calibration
            self._calibrate(
                model=self.decoder,
                tokenizer=data.tokenizer,
                event="prepare_pt2e",
                user_calibration_data=data.calibration_data.datasets,
                tok_embedding=self.tok_embedding,
                intermediate_outputs=image_embedding,
            )

            self.decoder = convert_pt2e(self.decoder)
            if self.apply_embedding:
                self.tok_embedding = convert_pt2e(self.tok_embedding)

            if self.control_args.verbose:
                if self.apply_embedding:
                    image_embedding = request.method_data[
                        VISION_ENCODER
                    ].calibration_data.qdq_intermediate_outputs[0]
                self._calibrate(
                    model=self.decoder,
                    tokenizer=data.tokenizer,
                    event="convert_pt2e",
                    user_calibration_data=data.calibration_data.datasets,
                    tok_embedding=self.tok_embedding,
                    intermediate_outputs=image_embedding,
                )

        # save logit's quantization attributes to meta
        self._save_logits_quant_attrs()

        # LLM: propagate kv cache quantization attributes for prefill model
        if not self.apply_embedding:
            if self.mode == self.Mode.DECODE:
                kv_quant_attrs, output_indices = {}, 0
                for node in self.decoder.graph.nodes:
                    if node.op == "output":
                        for output in node.args[0]:
                            kv_quant_attrs[output_indices] = output.args[1:]
                            output_indices += 1
                        break

                data.custom_annotation += (
                    partial(
                        annotate_prefill_kv_output,
                        kv_quant_attrs=kv_quant_attrs,
                    ),
                )
        # MultiModal: save kv cache IO quantization attributes to requant kv cache from prefill output scale/zero_point to decode input scale/zero_point
        else:
            # save input kv cache's quantization attributes to meta
            if self.mode == self.Mode.DECODE:
                self._save_input_kv_cache_quant_attrs()

            # save output kv cache's quantization attributes to meta
            if self.mode == self.Mode.PREFILL:
                self._save_output_kv_cache_quant_attrs()

        # setup quantized IO
        self.passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
        self.passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
            "get_quant_io_dtype_fn"
        ] = partial(self._tag_ios, fixed_point_type=fixed_point_type)
        if self.tok_embedding_passes_job is not None:
            self.tok_embedding_passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
            self.tok_embedding_passes_job[TagQuantIO][
                QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY
            ]["get_quant_io_dtype_fn"] = partial(
                self._tag_ios, fixed_point_type=fixed_point_type
            )


class HybridTextDecoder(Component):
    @log_info
    def __init__(
        self,
        control_args: argparse.Namespace,
        config: LLMModelConfig,
        apply_embedding: bool = False,
    ):
        self.decode = TextDecoder(
            control_args,
            config,
            TextDecoder.Mode.DECODE,
            apply_embedding=apply_embedding,
        )
        self.prefill = TextDecoder(
            control_args,
            config,
            TextDecoder.Mode.PREFILL,
            apply_embedding=apply_embedding,
        )
        self.control_args = control_args
        self.config = config
        self.set_next(self.decode).set_next(self.prefill)

        self.apply_embedding = apply_embedding

    @log_info
    def compile(self, request: Request):  # noqa: C901
        # force overriding frozen parameters here for model quantizing under seq mse scenario
        # this will make weight sharing work properly
        if self.config.seq_mse_candidates != 0 and self.control_args.model_mode != "kv":
            decode, prefill = self.decode.decoder, self.prefill.decoder
            override_nodes = {
                str(node.meta["nn_module_stack"].values()): node
                for node in prefill.graph.nodes
                if node.target == torch.ops.aten.conv2d.default
            }
            indices_map = {
                # (affine_tensor, group_size, scales, zero_points, dtype, min, max)
                torch.ops.torchao.dequantize_affine: [0, 2, 3],
                # (per_channel_tensor, scales, zero_points, dim, dtype, min, max)
                torch.ops.quantized_decomposed.dequantize_per_channel.default: [
                    0,
                    1,
                    2,
                ],
                # should not need to worry about per-tensor case
            }
            for node in decode.graph.nodes:
                if node.target == torch.ops.aten.conv2d.default:
                    if target_node := override_nodes.get(
                        str(node.meta["nn_module_stack"].values())
                    ):
                        # arguments of conv: (input, weight, bias)
                        for i, dq_node in enumerate(node.args[1:]):
                            for index in indices_map[dq_node.target]:
                                setattr(
                                    prefill,
                                    target_node.args[i + 1].args[index].target,
                                    getattr(decode, dq_node.args[index].target),
                                )
                    else:
                        raise RuntimeError("failed to override quantization attribute")

        # prepare lowering tok_embedding if applicable
        if self.apply_embedding:
            tok_embedding_data = request.method_data[TEXT_EMBEDDING]
            models = [
                d for d in [self.decode, self.prefill] if d.tok_embedding is not None
            ]
            tok_embedding_example_inputs = [
                m.tok_embedding_export_input for m in models if m is not None
            ]  # tokens
            tok_embedding_graph_names = TEXT_EMBEDDING_GRAPH_NAMES[: len(models)]

        # prepare lowering decoder
        data = request.method_data[TEXT_DECODER]
        models = [d for d in [self.decode, self.prefill] if d.decoder is not None]
        example_inputs = [m.export_input for m in models if m is not None]
        # For backward compatibility, we keep the graph name as forward if we use kv mode for evaluation LLM models
        graph_names = ["forward"] if len(models) == 1 else DECODER_GRAPH_NAMES

        # start lowering
        if self.apply_embedding:
            tok_embedding_edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                module=dict(
                    zip(
                        tok_embedding_graph_names,
                        [model.tok_embedding for model in models],
                    )
                ),
                inputs=dict(
                    zip(tok_embedding_graph_names, tok_embedding_example_inputs)
                ),
                compiler_specs=dict(
                    zip(tok_embedding_graph_names, tok_embedding_data.compile_spec)
                ),
                dep_table=dict(
                    zip(
                        tok_embedding_graph_names,
                        [model.tok_embedding_dep_table for model in models],
                    )
                ),
                passes_job=dict(
                    zip(
                        tok_embedding_graph_names,
                        [model.tok_embedding_passes_job for model in models],
                    )
                ),
            )
            if self.control_args.verbose:
                for ep in tok_embedding_edge_prog_mgr._edge_programs.values():
                    print_delegation_info(ep.graph_module)

            executorch_config = ExecutorchBackendConfig(
                # For shared buffer, user must pass the memory address
                # which is allocated by RPC memory to executor runner
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False,
                    alloc_graph_output=False,
                ),
            )
            tok_embedding_exec_prog_mgr = tok_embedding_edge_prog_mgr.to_executorch(
                executorch_config
            )
            data = request.method_data[TEXT_EMBEDDING]
            with open(
                f"{self.control_args.artifact}/{data.pte_filename}.pte", "wb"
            ) as file:
                tok_embedding_exec_prog_mgr.write_to_file(file)

        # decoder lowering
        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            module=dict(zip(graph_names, [model.decoder for model in models])),
            inputs=dict(zip(graph_names, example_inputs)),
            compiler_specs=dict(zip(graph_names, data.compile_spec)),
            constant_methods={**self.prefill.meta, **self.decode.meta},
            dep_table=dict(zip(graph_names, [model.dep_table for model in models])),
            passes_job=dict(zip(graph_names, [model.passes_job for model in models])),
            skip_node_op_set={"llama.fallback.default"},
        )

        if self.config.num_sharding > 1 and self.control_args.model_mode == "kv":
            # weight-sharing based context binaries cannot be opened in x86 host
            update_spill_fill_size(edge_prog_mgr.exported_program())

        if self.control_args.verbose:
            for ep in edge_prog_mgr._edge_programs.values():
                print_delegation_info(ep.graph_module)

        executorch_config = ExecutorchBackendConfig(
            # For shared buffer, user must pass the memory address
            # which is allocated by RPC memory to executor runner
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                alloc_graph_output=False,
            ),
        )
        exec_prog_mgr = edge_prog_mgr.to_executorch(executorch_config)
        data = request.method_data[TEXT_DECODER]
        with open(
            f"{self.control_args.artifact}/{data.pte_filename}.pte", "wb"
        ) as file:
            exec_prog_mgr.write_to_file(file)


class Modality(Component):
    def __init__(
        self, control_args: argparse.Namespace, config: LLMModelConfig, modality
    ):
        self.control_args = control_args
        self.model = None
        self.modality = modality
        repo_id = config.repo_id

        if config := getattr(config, modality, None):
            if modality == TEXT_ENCODER or modality == AUDIO_ENCODER:
                raise NotImplementedError(f"{modality} is under development")

            auto_model = AutoModel.from_pretrained(
                repo_id, _attn_implementation="eager"
            )
            # Create an instance of the config class since it has init=False
            self.model = config().create_encoder(auto_model.config)
            # set strict to false to simplify parameter loading for non-text models
            auto_model = auto_model.eval()
            self.model = self.model.eval()
            self.model.load_state_dict(auto_model.state_dict(), strict=False)
            self.example_input = self.model.get_example_inputs()
            self.preprocess = self.model.preprocess

            # set quant recipe
            self.quant_recipe: EncoderQuantRecipe = (
                config.quant_recipe(True) if config.quant_recipe else None
            )

    def compile(self, request: Request):
        if self.model is None:
            return

        request_data = request.method_data[self.modality]
        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            module=self.model,
            inputs=self.example_input,
            compiler_specs=request_data.compile_spec,
        )
        if self.control_args.verbose:
            print_delegation_info(edge_prog_mgr.exported_program().graph_module)

        exec_prog_mgr = edge_prog_mgr.to_executorch(ExecutorchBackendConfig())
        data = request.method_data[self.modality]
        with open(
            f"{self.control_args.artifact}/{data.pte_filename}.pte", "wb"
        ) as file:
            exec_prog_mgr.write_to_file(file)

    def quantize(self, request: Request):
        if self.model is None or self.quant_recipe is None:
            return

        request_data = request.method_data[self.modality]
        with torch.no_grad():
            self.model = torch.export.export(self.model, self.example_input).module()

            quantizer = make_quantizer()
            quantizer.recipe = self.quant_recipe
            self.model = prepare_pt2e(self.model, quantizer)

            # calibration
            intermediate_outputs = []
            for data in request_data.calibration_data.datasets:
                output = self.model(*self.preprocess(data))
                intermediate_outputs.append(
                    (output,) if isinstance(output, torch.Tensor) else output
                )
            # update intermediate outputs for next modality
            request_data.calibration_data.intermediate_outputs = intermediate_outputs

            self.model = convert_pt2e(self.model)

            qdq_intermediate_outputs = []
            if self.control_args.verbose:
                for data in request_data.calibration_data.datasets:
                    output = self.model(*self.preprocess(data))
                    qdq_intermediate_outputs.append(
                        (output,) if isinstance(output, torch.Tensor) else output
                    )
                # update qdq intermediate outputs for next modality
                request_data.calibration_data.qdq_intermediate_outputs = (
                    qdq_intermediate_outputs
                )


class MultiModalManager(Component):
    def __init__(self, control_args: argparse.Namespace, config: LLMModelConfig):
        self.audio_encoder = Modality(
            control_args,
            config,
            AUDIO_ENCODER,
        )
        self.text_encoder = Modality(
            control_args,
            config,
            TEXT_ENCODER,
        )
        self.vision_encoder = Modality(
            control_args,
            config,
            VISION_ENCODER,
        )
        self.text_decoder = HybridTextDecoder(
            control_args,
            config,
            apply_embedding=self.audio_encoder.model or self.vision_encoder.model,
        )
        self._modalities = [
            AUDIO_ENCODER,
            TEXT_ENCODER,
            VISION_ENCODER,
            TEXT_EMBEDDING,
            TEXT_DECODER,
        ]
        # build dependency chain
        self.set_next(self.vision_encoder).set_next(self.audio_encoder).set_next(
            self.text_decoder
        )

    def process(self, request: Request) -> Request:
        Processor.process(self, request)

    @log_info
    def compile(
        self,
        compile_specs: Dict[str, List[CompileSpec]],
        pte_filenames: Dict[str, str],
    ):
        compile_request = Request(
            inspect.currentframe().f_code.co_name,
            {
                m: Request.Data(
                    compile_spec=compile_specs[m],
                    pte_filename=pte_filenames[m],
                )
                for m in self._modalities
            },
        )
        self.process(compile_request)

    @log_info
    def quantize(
        self,
        calibration_data: Dict[str, List[Any]],
        tokenizer,
    ):
        quantize_request = Request(
            inspect.currentframe().f_code.co_name,
            {
                m: Request.Data(
                    calibration_data=Request.CalibrationData(
                        datasets=calibration_data[m]
                    ),
                    tokenizer=tokenizer,
                )
                for m in self._modalities
            },
        )
        self.process(quantize_request)
