# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from abc import ABC
from dataclasses import dataclass
from typing import Callable, Dict, Type

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

BASE_DIR = os.path.dirname(__file__)


@dataclass(init=False, frozen=True)
class HFModel(ABC):
    """Base class for all hugging face models

    repo_id: Hugging Face Repo ID.
    params_path: Path to model's config.json. If the corresponding .json has not yet exsit, please create one.
    convert_weights: Used to convert Hugging Face weights parameters to Static Decoder's parameter naming.
    transform_weight: Set to true to change HuggingFace weight to improve the performance of RoPE in HTP backend.
    instruct_model: True if the model uses chat templates. Check Hugging Face model card to ensure the model uses chat templates.
    """

    repo_id: str
    params_path: str
    convert_weights: Callable
    transform_weight: bool
    instruct_model: bool


SUPPORTED_HF_MODELS: Dict[str, HFModel] = {}


def register_hf_model(name: str):
    def decorator(cls: Type[HFModel]):
        SUPPORTED_HF_MODELS[name.lower()] = cls()
        return cls()

    return decorator


@register_hf_model("qwen2_5-0_5b")
@dataclass(init=False, frozen=True)
class Qwen2_5_0_5B(HFModel):
    repo_id: str = "Qwen/Qwen2.5-0.5B"
    params_path: str = os.path.join(
        BASE_DIR, "../../../models/qwen2_5/config/0_5b_config.json"
    )
    convert_weights = convert_qwen2_5_weights
    transform_weight = False
    instruct_model = False


@register_hf_model("qwen2_5-1_5b")
@dataclass(init=False, frozen=True)
class Qwen2_5_1_5B(HFModel):
    repo_id: str = "Qwen/Qwen2.5-1.5B"
    params_path: str = os.path.join(
        BASE_DIR, "../../../models/qwen2_5/config/1_5b_config.json"
    )
    convert_weights = convert_qwen2_5_weights
    transform_weight = False
    instruct_model = False


@register_hf_model("qwen3-0_6b")
@dataclass(init=False, frozen=True)
class Qwen3_0_6B(HFModel):
    repo_id: str = "Qwen/Qwen3-0.6B"
    params_path: str = os.path.join(
        BASE_DIR, "../../../models/qwen3/config/0_6b_config.json"
    )
    convert_weights = convert_qwen3_weights
    transform_weight = False
    instruct_model = True


@register_hf_model("qwen3-1_7b")
@dataclass(init=False, frozen=True)
class Qwen3_1_7B(HFModel):
    repo_id: str = "Qwen/Qwen3-1.7B"
    params_path: str = os.path.join(
        BASE_DIR, "../../../models/qwen3/config/1_7b_config.json"
    )
    convert_weights = convert_qwen3_weights
    transform_weight = False
    instruct_model = True


@register_hf_model("phi_4_mini")
@dataclass(init=False, frozen=True)
class Phi4Mini(HFModel):
    repo_id: str = "microsoft/Phi-4-mini-instruct"
    params_path: str = os.path.join(
        BASE_DIR, "../../../models/phi_4_mini/config/config.json"
    )
    convert_weights = convert_phi_4_mini_weights
    transform_weight = False
    instruct_model = True


@register_hf_model("smollm2_135m")
@dataclass(init=False, frozen=True)
class Smollm2_135M(HFModel):
    repo_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    params_path: str = os.path.join(
        BASE_DIR, "../../../models/smollm2/135M_config.json"
    )
    convert_weights = convert_smollm2_weights
    transform_weight = True
    instruct_model = True
