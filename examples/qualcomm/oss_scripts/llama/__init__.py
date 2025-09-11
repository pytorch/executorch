# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Callable, Dict, Tuple, Type

import torch
from executorch.backends.qualcomm.quantizer.custom_annotation import (
    annotate_down_proj,
    annotate_kv_8bit,
    annotate_output_16a8w,
    annotate_qkv_proj_sha,
    StaticLLMQuantConfig,
)
from executorch.backends.qualcomm.quantizer.qconfig import (
    get_ptq_per_channel_quant_config,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype

from executorch.examples.models.gemma3 import convert_weights as convert_gemma3_weights
from executorch.examples.models.phi_4_mini import (
    convert_weights as convert_phi_4_mini_weights,
)
from executorch.examples.models.qwen2_5 import (
    convert_weights as convert_qwen2_5_weights,
)
from executorch.examples.models.qwen3 import convert_weights as convert_qwen3_weights
from executorch.examples.models.smollm2 import (
    convert_weights as convert_smollm2_weights,
)
from executorch.examples.models.smollm3 import (
    convert_weights as convert_smollm3_weights,
)

from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    DECODER_MODEL_VERSION,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    MultiScopeAwareLlamaModel,
)
from tabulate import tabulate
from torchao.quantization.pt2e import MinMaxObserver


BASE_DIR = os.path.dirname(__file__)


LLM_VARIANT_ARCHS = {
    "gemma3-1b": MultiScopeAwareLlamaModel,
}
annotate_wqkv_sha = partial(
    annotate_qkv_proj_sha,
    qkv_tags={
        StaticLLMQuantConfig.wq_sha,
        StaticLLMQuantConfig.wk_sha,
        StaticLLMQuantConfig.wv_sha,
    },
)
annotate_wv_sha = partial(annotate_qkv_proj_sha, qkv_tags={StaticLLMQuantConfig.wv_sha})


@dataclass(init=False, frozen=True)
class LLMModelConfig(ABC):
    """
    Base class for all LLM Models, including native Llama and Hugging Face Models.
    This class stores configs for each supported LLM model, including quant config

    repo_id: Hugging Face Repo ID.
    params_path: Path to model's config.json. If the corresponding .json has not yet exist, please create one.
    convert_weights: Used to convert Hugging Face weights parameters to Static Decoder's parameter naming.
    decoder_model_version: This is to determine the chat template to use during runtime(qnn_llama_runner).
    transform_weight: Set to true to change Hugging Face weight to improve the performance of RoPE in HTP backend.
    instruct_model: True if the model uses chat templates. Check Hugging Face model card to ensure the model uses chat templates.
    num_sharding: Specify the number of splits by inserting the fallback custom op. The graph will be split evenly by layers.
    ptq: Set to true to perform PTQ quantization. Support 16a16w, 16a8w, 16a4w, 16a4w_block, 8a8w.
    group_size: Group size used in block quantization for weight quantization. Will only be used when ptq = 16a4w_block
    masked_softmax: The MaskedSoftmax feature is designed to optimize the LLMs accuracy and performance executed on HTP backend.
                    MaskedSoftmax is used to replace the Softmax(Add(In, Mask)) structure in attention block in LLMs during backend optimization.
                    For more details, please refer to QNN documents. Note that it is only supported starting from QNN 2.35.
    seq_mse_candidates: Number of steps to sequentially search for optimum scales for quantized parameters which will minimize
                        the MSE of activation value between floating point golden & fake quantization.
    r1: Enable SpinQuant R1 quantization optimization.
    r2: Enable SpinQuant R2 quantization optimization.
    r3: Enable SpinQuant R3 quantization optimization.
    custom_annotation: Custom annotation to use when setting quant configs for the model.
    """

    repo_id: str
    params_path: str
    convert_weights: Callable
    # TODO: Replace decoder_model_version with chat_template.jinja
    decoder_model_version: str
    transform_weight: bool
    instruct_model: bool
    num_sharding: int
    ptq: QuantDtype
    group_size: int
    masked_softmax: bool
    seq_mse_candidates: int
    r1: bool
    r2: bool
    r3: bool
    custom_annotation: Tuple

    def __str__(self):  # noqa: C901
        """
        Visualize the current LLMModelConfig settings in a readable table format.

        This method helps users quickly inspect key configuration,
        skipping internal or irrelevant attributes and formatting complex types
        like functions, enums, and partials for clarity.

        Returns:
            str: A table showing the current config for LLM models.
        """

        def format_value(v):
            if isinstance(v, partial):
                func_name = (
                    v.func.__name__ if hasattr(v.func, "__name__") else str(v.func)
                )
                return f"partial({func_name})"
            elif isinstance(v, Callable):
                return v.__name__ if hasattr(v, "__name__") else str(v)
            elif isinstance(v, Enum):
                return f"{v.__class__.__name__}.{v.name}"
            elif isinstance(v, (tuple, list)):
                return "(" + ", ".join(format_value(i) for i in v) + ")"
            elif isinstance(v, (str, int, float, bool)):
                return v
            else:
                return f"<{v.__class__.__name__}>"

        attrs = {}
        for k in dir(self):
            if k.startswith("_") or k in {"convert_weights", "params_path"}:
                continue
            try:
                v = getattr(self, k)
                if k in {"get_kv_io_bit_width", "get_logits_output_bit_width"}:
                    v = v()
            except Exception:
                v = f"Warning: failed to retrieve config for '{k}'"
            if isinstance(v, (str, int, float, bool, tuple, list, Callable)):
                attrs[k] = format_value(v)
        table = [(k, v) for k, v in attrs.items()]
        return tabulate(table, headers=["Config", "Value"], tablefmt="grid")

    def get_kv_io_bit_width(self) -> int:
        if self.ptq is None:
            return 32
        elif (
            self.ptq == QuantDtype.use_8a8w
            or annotate_kv_8bit in self.custom_annotation
        ):
            return 8
        else:
            # If quantized but not 8a8w or mix_quantization, it has to be 16bit kv io.
            return 16

    def get_logits_output_bit_width(self) -> int:
        # We use 16bit logits for all quant config
        return 32 if self.ptq is None else 16


SUPPORTED_LLM_MODELS: Dict[str, LLMModelConfig] = {}


def register_llm_model(name: str):
    def decorator(cls: Type[LLMModelConfig]):
        cls.decoder_model_version = DECODER_MODEL_VERSION[name]
        SUPPORTED_LLM_MODELS[name.lower()] = cls()
        return cls()

    return decorator


@register_llm_model("stories260k")
@dataclass(init=False, frozen=True)
class LlamaStories260K(LLMModelConfig):
    repo_id = None
    params_path = None
    convert_weights = None
    transform_weight = True
    instruct_model = False

    num_sharding = 1
    # quant config
    ptq = QuantDtype.use_16a4w
    group_size = None
    masked_softmax = False
    seq_mse_candidates = 0
    r1 = False
    r2 = False
    r3 = False
    quantization_config_wv_sha_8a4w = get_ptq_per_channel_quant_config(
        act_dtype=torch.uint8,
        weight_dtype=torch.int4,
        act_observer=MinMaxObserver,
        act_symmetric=True,
    )
    custom_annotation = (
        annotate_kv_8bit,
        annotate_output_16a8w,
        partial(annotate_wv_sha, quantization_config=quantization_config_wv_sha_8a4w),
    )


@register_llm_model("stories110m")
@dataclass(init=False, frozen=True)
class LlamaStories110M(LLMModelConfig):
    repo_id = None
    params_path = None
    convert_weights = None
    transform_weight = True
    instruct_model = False

    num_sharding = 1
    # quant config
    ptq = QuantDtype.use_16a4w
    group_size = None
    masked_softmax = False
    seq_mse_candidates = 0
    r1 = False
    r2 = False
    r3 = False
    quantization_config_wv_sha_8a4w = get_ptq_per_channel_quant_config(
        act_dtype=torch.uint8,
        weight_dtype=torch.int4,
        act_observer=MinMaxObserver,
        act_symmetric=True,
    )
    custom_annotation = (
        annotate_kv_8bit,
        annotate_output_16a8w,
        partial(annotate_wv_sha, quantization_config=quantization_config_wv_sha_8a4w),
    )


@register_llm_model("llama3_2-1b_instruct")
@dataclass(init=False, frozen=True)
class Llama3_2_1B_Instruct(LLMModelConfig):
    repo_id = None
    params_path = None
    convert_weights = None
    transform_weight = True
    # The Llama3_2 enabled should be instruct, however, Llama's tokenizer does not provide utility to apply chat template.
    instruct_model = False

    num_sharding = 1
    # quant config
    ptq = QuantDtype.use_16a4w_block
    group_size = 32
    masked_softmax = False
    seq_mse_candidates = 1000
    r1 = False
    r2 = False
    r3 = False
    quantization_config_down_proj_16a8w = get_ptq_per_channel_quant_config(
        torch.uint16, weight_dtype=torch.int8, act_observer=MinMaxObserver
    )
    custom_annotation = (
        annotate_kv_8bit,
        annotate_output_16a8w,
        partial(
            annotate_down_proj, quantization_config=quantization_config_down_proj_16a8w
        ),
    )


@register_llm_model("llama3_2-3b_instruct")
@dataclass(init=False, frozen=True)
class Llama3_2_3B_Instruct(LLMModelConfig):
    repo_id = None
    params_path = None
    convert_weights = None
    transform_weight = True
    # The Llama3_2 enabled should be instruct, however, Llama's tokenizer does not provide utility to apply chat template.
    instruct_model = False

    num_sharding = 4
    # quant config
    ptq = QuantDtype.use_16a4w_block
    group_size = 32
    masked_softmax = False
    seq_mse_candidates = 0
    r1 = False
    r2 = False
    r3 = False
    custom_annotation = (
        annotate_kv_8bit,
        annotate_output_16a8w,
    )


@register_llm_model("gemma3-1b")
@dataclass(init=False, frozen=True)
class Gemma3(LLMModelConfig):
    repo_id: str = "google/gemma-3-1b-it"
    params_path: str = os.path.join(
        BASE_DIR, "../../../models/gemma3/config/1b_config.json"
    )
    convert_weights = convert_gemma3_weights
    transform_weight = False
    instruct_model = True

    num_sharding = 1
    # quant config
    ptq = QuantDtype.use_16a4w_block
    group_size = 64
    masked_softmax = True
    seq_mse_candidates = 0
    r1 = False
    r2 = False
    r3 = False
    quantization_config_wv_sha_16a8w = get_ptq_per_channel_quant_config(
        torch.uint16, weight_dtype=torch.int8, act_observer=MinMaxObserver
    )
    custom_annotation = (
        annotate_kv_8bit,
        partial(annotate_wv_sha, quantization_config=quantization_config_wv_sha_16a8w),
    )


@register_llm_model("phi_4_mini")
@dataclass(init=False, frozen=True)
class Phi4Mini(LLMModelConfig):
    repo_id: str = "microsoft/Phi-4-mini-instruct"
    params_path: str = os.path.join(
        BASE_DIR, "../../../models/phi_4_mini/config/config.json"
    )
    convert_weights = convert_phi_4_mini_weights
    transform_weight = False
    instruct_model = True

    num_sharding = 8
    # quant config
    ptq = QuantDtype.use_16a4w_block
    group_size = 16
    masked_softmax = False
    seq_mse_candidates = 0
    r1 = False
    r2 = False
    r3 = False
    quantization_config_wv_sha_8a4w = get_ptq_per_channel_quant_config(
        act_dtype=torch.uint8,
        weight_dtype=torch.int4,
        act_observer=MinMaxObserver,
        act_symmetric=True,
    )
    custom_annotation = (
        annotate_kv_8bit,
        annotate_output_16a8w,
        partial(annotate_wv_sha, quantization_config=quantization_config_wv_sha_8a4w),
    )


@register_llm_model("qwen2_5-0_5b")
@dataclass(init=False, frozen=True)
class Qwen2_5_0_5B(LLMModelConfig):
    repo_id: str = "Qwen/Qwen2.5-0.5B"
    params_path: str = os.path.join(
        BASE_DIR, "../../../models/qwen2_5/config/0_5b_config.json"
    )
    convert_weights = convert_qwen2_5_weights
    transform_weight = False
    instruct_model = False

    num_sharding = 1
    # quant config
    ptq = QuantDtype.use_16a4w_block
    group_size = 16
    masked_softmax = True
    seq_mse_candidates = 0
    r1 = False
    r2 = False
    r3 = True
    custom_annotation = ()


@register_llm_model("qwen2_5-1_5b")
@dataclass(init=False, frozen=True)
class Qwen2_5_1_5B(LLMModelConfig):
    repo_id: str = "Qwen/Qwen2.5-1.5B"
    params_path: str = os.path.join(
        BASE_DIR, "../../../models/qwen2_5/config/1_5b_config.json"
    )
    convert_weights = convert_qwen2_5_weights
    transform_weight = False
    instruct_model = False

    num_sharding = 1
    # quant config
    ptq = QuantDtype.use_16a4w_block
    group_size = 16
    masked_softmax = True
    seq_mse_candidates = 0
    r1 = False
    r2 = False
    r3 = True
    custom_annotation = (annotate_output_16a8w,)


@register_llm_model("qwen3-0_6b")
@dataclass(init=False, frozen=True)
class Qwen3_0_6B(LLMModelConfig):
    repo_id: str = "Qwen/Qwen3-0.6B"
    params_path: str = os.path.join(
        BASE_DIR, "../../../models/qwen3/config/0_6b_config.json"
    )
    convert_weights = convert_qwen3_weights
    transform_weight = False
    instruct_model = True

    num_sharding = 1
    # quant config
    ptq = QuantDtype.use_16a4w_block
    group_size = 32
    masked_softmax = True
    seq_mse_candidates = 1000
    r1 = False
    r2 = False
    r3 = False
    quantization_config_down_proj_16a8w = get_ptq_per_channel_quant_config(
        torch.uint16, weight_dtype=torch.int8, act_observer=MinMaxObserver
    )
    custom_annotation = (
        partial(
            annotate_down_proj, quantization_config=quantization_config_down_proj_16a8w
        ),
    )


@register_llm_model("qwen3-1_7b")
@dataclass(init=False, frozen=True)
class Qwen3_1_7B(LLMModelConfig):
    repo_id: str = "Qwen/Qwen3-1.7B"
    params_path: str = os.path.join(
        BASE_DIR, "../../../models/qwen3/config/1_7b_config.json"
    )
    convert_weights = convert_qwen3_weights
    transform_weight = False
    instruct_model = True

    num_sharding = 1
    # quant config
    ptq = QuantDtype.use_16a4w_block
    group_size = 16
    masked_softmax = True
    seq_mse_candidates = 0
    r1 = False
    r2 = False
    r3 = True
    custom_annotation = (
        annotate_kv_8bit,
        annotate_output_16a8w,
    )


@register_llm_model("smollm2_135m")
@dataclass(init=False, frozen=True)
class Smollm2_135M(LLMModelConfig):
    repo_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    params_path: str = os.path.join(
        BASE_DIR, "../../../models/smollm2/135M_config.json"
    )
    convert_weights = convert_smollm2_weights
    transform_weight = True
    instruct_model = True

    num_sharding = 1
    # quant config
    ptq = QuantDtype.use_16a8w
    group_size = None
    masked_softmax = False
    seq_mse_candidates = 0
    r1 = False
    r2 = False
    r3 = False
    custom_annotation = ()


@register_llm_model("smollm3-3b")
@dataclass(init=False, frozen=True)
class Smollm3_3B(LLMModelConfig):
    repo_id: str = "HuggingFaceTB/SmolLM3-3B"
    params_path: str = os.path.join(BASE_DIR, "../../../models/smollm3/3b_config.json")
    convert_weights = convert_smollm3_weights
    transform_weight = False
    instruct_model = True

    num_sharding = 4
    # quant config
    ptq = QuantDtype.use_16a4w_block
    group_size = 32
    masked_softmax = True
    r1 = False
    r2 = False
    r3 = False
    quantization_config_wqkv_sha_16a8w = get_ptq_per_channel_quant_config(
        torch.uint16, weight_dtype=torch.int8, act_observer=MinMaxObserver
    )
    custom_annotation = (
        annotate_kv_8bit,
        annotate_output_16a8w,
        partial(
            annotate_wqkv_sha, quantization_config=quantization_config_wqkv_sha_16a8w
        ),
    )
