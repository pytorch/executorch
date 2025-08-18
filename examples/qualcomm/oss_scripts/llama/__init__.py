# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from abc import ABC
from dataclasses import dataclass, field
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
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    DECODER_MODEL_VERSION,
)

BASE_DIR = os.path.dirname(__file__)


@dataclass(init=False, frozen=True)
class HFModel(ABC):
    repo_id: str
    params_path: str
    runner_version: str
    convert_weights: Callable


SUPPORTED_HF_MODELS: Dict[str, Type[HFModel]] = {}


def register_hf_model(name: str):
    def decorator(cls: Type[HFModel]):
        SUPPORTED_HF_MODELS[name.lower()] = cls()
        return cls()

    return decorator


@register_hf_model("qwen2_5")
@dataclass(init=False, frozen=True)
class Qwen2_5(HFModel):
    repo_id: str = "Qwen/Qwen2.5-0.5B"
    params_path: str = os.path.join(
        BASE_DIR, "../../../models/qwen2_5/config/0_5b_config.json"
    )
    runner_version: str = field(default=DECODER_MODEL_VERSION["qwen2_5"])
    convert_weights = convert_qwen2_5_weights
    transform_weight = False


@register_hf_model("qwen3_0_6b")
@dataclass(init=False, frozen=True)
class Qwen3_0_6B(HFModel):
    repo_id: str = "Qwen/Qwen3-0.6B"
    params_path: str = os.path.join(
        BASE_DIR, "../../../models/qwen3/config/0_6b_config.json"
    )
    runner_version: str = field(default=DECODER_MODEL_VERSION["qwen2_5"])
    convert_weights = convert_qwen3_weights
    transform_weight = False


@register_hf_model("qwen3_1_7b")
@dataclass(init=False, frozen=True)
class Qwen3_1_7B(HFModel):
    repo_id: str = "Qwen/Qwen3-1.7B"
    params_path: str = os.path.join(
        BASE_DIR, "../../../models/qwen3/config/1_7b_config.json"
    )
    runner_version: str = field(default=DECODER_MODEL_VERSION["qwen2_5"])
    convert_weights = convert_qwen3_weights
    transform_weight = False


@register_hf_model("phi_4_mini")
@dataclass(init=False, frozen=True)
class Phi4Mini(HFModel):
    repo_id: str = "microsoft/Phi-4-mini-instruct"
    params_path: str = os.path.join(
        BASE_DIR, "../../../models/phi_4_mini/config/config.json"
    )
    runner_version: str = field(default=DECODER_MODEL_VERSION["phi_4_mini"])
    convert_weights = convert_phi_4_mini_weights
    transform_weight = False


@register_hf_model("smollm2_135m")
@dataclass(init=False, frozen=True)
class Smollm2_135M(HFModel):
    repo_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    params_path: str = os.path.join(
        BASE_DIR, "../../../models/smollm2/135M_config.json"
    )
    runner_version: str = field(default=DECODER_MODEL_VERSION["smollm2_135m"])
    convert_weights = convert_smollm2_weights
    transform_weight = True
