# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Tuple

import pytest
import torch
from executorch.backends.arm._passes import (
    ConvertInt64ConstOpsToInt32Pass,
    ConvertInt64OutputOpsToInt32Pass,
    InsertInt32CastsAfterInt64PlaceholdersPass,
)

from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)

pytest.importorskip("transformers.models.phi3")

from executorch.backends.arm.test.models.phi3.phi3_module_test_configs import (
    get_phi3_test_config,
)
from transformers.models.phi3.configuration_phi3 import Phi3Config  # noqa: E402
from transformers.models.phi3.modeling_phi3 import (  # noqa: E402
    Phi3Attention,
    Phi3DecoderLayer,
    Phi3MLP,
    Phi3RMSNorm,
    Phi3RotaryEmbedding,
)

input_t1 = Tuple[torch.Tensor]
input_t2 = Tuple[torch.Tensor, torch.Tensor]


def _phi3_config() -> Phi3Config:
    return get_phi3_test_config()


def _hidden_states(
    config: Phi3Config, dtype: torch.dtype, batch: int = 2, seq: int = 4
) -> torch.Tensor:
    hidden_size = config.hidden_size
    if hidden_size is None:
        raise RuntimeError("Phi3Config hidden_size must be set for test inputs.")
    return torch.randn(batch, seq, hidden_size, dtype=dtype)


def _position_ids(batch: int = 2, seq: int = 4) -> torch.Tensor:
    return torch.arange(seq, dtype=torch.long).unsqueeze(0).repeat(batch, 1)


class Phi3AttentionModule(torch.nn.Module):
    def __init__(self, config: Phi3Config) -> None:
        super().__init__()
        self.attn = Phi3Attention(config, layer_idx=0)
        self.rotary = Phi3RotaryEmbedding(config)

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        position_embeddings = self.rotary(hidden_states, position_ids)
        return self.attn(hidden_states, position_embeddings, None)[0]


class Phi3DecoderLayerModule(torch.nn.Module):
    def __init__(self, config: Phi3Config) -> None:
        super().__init__()
        self.layer = Phi3DecoderLayer(config, layer_idx=0)
        self.rotary = Phi3RotaryEmbedding(config)

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        position_embeddings = self.rotary(hidden_states, position_ids)
        output, _ = self.layer(hidden_states, position_embeddings=position_embeddings)
        return output


def _module_cases() -> list[
    tuple[
        str,
        Callable[[Phi3Config], torch.nn.Module],
        Callable[[Phi3Config, torch.dtype], Tuple],
    ]
]:
    return [
        (
            "rms_norm",
            lambda cfg: Phi3RMSNorm(
                cfg.hidden_size,
                eps=float(cfg.rms_norm_eps) if cfg.rms_norm_eps is not None else 1e-6,
            ),
            lambda cfg, dtype: (_hidden_states(cfg, dtype),),
        ),
        (
            "mlp",
            lambda cfg: Phi3MLP(cfg),
            lambda cfg, dtype: (_hidden_states(cfg, dtype),),
        ),
        (
            "attention",
            lambda cfg: Phi3AttentionModule(cfg),
            lambda cfg, dtype: (
                _hidden_states(cfg, dtype),
                _position_ids(seq=min(4, cfg.max_position_embeddings or 4)),
            ),
        ),
        (
            "decoder_layer",
            lambda cfg: Phi3DecoderLayerModule(cfg),
            lambda cfg, dtype: (
                _hidden_states(cfg, dtype),
                _position_ids(seq=min(4, cfg.max_position_embeddings or 4)),
            ),
        ),
    ]


def _module_cases_int() -> list[object]:
    xfail_reason = (
        "INT8 TOSA path delegates to executorch_call_delegate for attention and "
        "decoder_layer (check_count.exir fails)."
    )
    return [
        (
            "rms_norm",
            lambda cfg: Phi3RMSNorm(
                cfg.hidden_size,
                eps=float(cfg.rms_norm_eps) if cfg.rms_norm_eps is not None else 1e-6,
            ),
            lambda cfg, dtype: (_hidden_states(cfg, dtype),),
        ),
        (
            "mlp",
            lambda cfg: Phi3MLP(cfg),
            lambda cfg, dtype: (_hidden_states(cfg, dtype),),
        ),
        pytest.param(
            "attention",
            lambda cfg: Phi3AttentionModule(cfg),
            lambda cfg, dtype: (
                _hidden_states(cfg, dtype),
                _position_ids(seq=min(4, cfg.max_position_embeddings or 4)),
            ),
            marks=pytest.mark.xfail(strict=True, reason=xfail_reason),
            id="attention",
        ),
        pytest.param(
            "decoder_layer",
            lambda cfg: Phi3DecoderLayerModule(cfg),
            lambda cfg, dtype: (
                _hidden_states(cfg, dtype),
                _position_ids(seq=min(4, cfg.max_position_embeddings or 4)),
            ),
            marks=pytest.mark.xfail(strict=True, reason=xfail_reason),
            id="decoder_layer",
        ),
    ]


def _dtype_cases() -> list:
    return [
        pytest.param(torch.float32, [], id="fp32"),
        pytest.param(
            torch.bfloat16,
            ["bf16"],
            id="bf16",
        ),
        pytest.param(
            torch.float16,
            [],
            id="fp16",
        ),
    ]


def _vgf_dtype_cases() -> list:
    return [
        pytest.param(torch.float32, id="fp32"),
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.xfail(reason="BF16 runtime support not ready for VGF."),
            id="bf16",
        ),
    ]


@pytest.mark.parametrize("dtype,tosa_extensions", _dtype_cases())
@pytest.mark.parametrize("name,module_factory,input_factory", _module_cases())
def test_phi3_tosa_FP_layers(
    dtype, tosa_extensions, name, module_factory, input_factory
):
    config = _phi3_config()
    module = module_factory(config).to(dtype)
    inputs = input_factory(config, dtype)
    atol = 1e-02 if dtype == torch.bfloat16 else 1e-03
    rtol = 1e-02 if dtype == torch.bfloat16 else 1e-03

    pipeline = TosaPipelineFP[input_t1 if len(inputs) == 1 else input_t2](
        module,
        inputs,
        aten_op=[],
        tosa_extensions=tosa_extensions or None,
        atol=atol,
        rtol=rtol,
        transform_passes=[
            ConvertInt64ConstOpsToInt32Pass(),
            ConvertInt64OutputOpsToInt32Pass(),
            InsertInt32CastsAfterInt64PlaceholdersPass(),
        ],
    )
    pipeline.run()


@pytest.mark.parametrize("name,module_factory,input_factory", _module_cases_int())
def test_phi3_tosa_INT_layers(name, module_factory, input_factory):
    config = _phi3_config()
    module = module_factory(config)
    inputs = input_factory(config, torch.float32)

    pipeline = TosaPipelineINT[input_t1 if len(inputs) == 1 else input_t2](
        module,
        inputs,
        aten_op=[],
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@pytest.mark.parametrize("dtype", _vgf_dtype_cases())
@pytest.mark.parametrize("name,module_factory,input_factory", _module_cases())
def test_phi3_vgf_no_quant_layers(name, module_factory, input_factory, dtype):
    config = _phi3_config()
    module = module_factory(config).to(dtype)
    inputs = input_factory(config, dtype)

    pipeline = VgfPipeline[input_t1 if len(inputs) == 1 else input_t2](
        module,
        inputs,
        aten_op=[],
        transform_passes=[
            ConvertInt64ConstOpsToInt32Pass(),
            ConvertInt64OutputOpsToInt32Pass(),
            InsertInt32CastsAfterInt64PlaceholdersPass(),
        ],
        quantize=False,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@pytest.mark.parametrize("dtype", _vgf_dtype_cases())
@pytest.mark.parametrize("name,module_factory,input_factory", _module_cases())
def test_phi3_vgf_quant_layers(name, module_factory, input_factory, dtype):
    config = _phi3_config()
    module = module_factory(config).to(dtype)
    inputs = input_factory(config, dtype)

    pipeline = VgfPipeline[input_t1 if len(inputs) == 1 else input_t2](
        module,
        inputs,
        aten_op=[],
        quantize=True,
    )
    pipeline.run()
