# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest

import torch
from executorch.backends.arm._passes.insert_int32_casts_after_int64_placeholders import (
    InsertInt32CastsAfterInt64PlaceholdersPass,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from transformers import LlamaConfig
from transformers.cache_utils import StaticCache

input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


test_configs = {
    "multihead_attention": LlamaConfig(num_attention_heads=32),
    "grouped_query_attention": LlamaConfig(
        num_attention_heads=32, num_key_value_heads=4
    ),
    "multi_query_attention": LlamaConfig(num_attention_heads=32, num_key_value_heads=1),
}


@torch.no_grad()
class StaticCacheModule(torch.nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        max_cache_len: int = 10,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        assert dtype is torch.float32

        self.config = config
        self.dtype = dtype

        self.cache = StaticCache(config=self.config, max_cache_len=max_cache_len)

        hidden_size = self.config.hidden_size
        num_attention_heads = self.config.num_attention_heads
        assert hidden_size is not None and num_attention_heads is not None

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        self.cache.early_initialization(
            1,
            self.num_attention_heads,
            self.hidden_size // self.num_attention_heads,
            self.dtype,
            torch.device("cpu"),
        )

        for i in range(len(self.cache.layers)):
            self.register_buffer(f"cache_layer_keys_{i}", self.cache.layers[i].keys)
            self.register_buffer(f"cache_layer_values_{i}", self.cache.layers[i].values)

    def forward(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key, value = self.cache.update(
            key_states,
            value_states,
            layer_idx=0,
            cache_kwargs={
                "cache_position": cache_position,
            },
        )

        return key.clone(), value.clone()

    def get_inputs(self) -> input_t:
        key_states = torch.randn(
            (
                1,
                self.num_attention_heads,
                1,
                self.hidden_size // self.num_attention_heads,
            ),
            dtype=torch.float32,
        )
        value_states = torch.randn(
            (
                1,
                self.num_attention_heads,
                1,
                self.hidden_size // self.num_attention_heads,
            ),
            dtype=torch.float32,
        )
        cache_position = torch.tensor([1], dtype=torch.int64)

        return key_states, value_states, cache_position


@common.parametrize("test_data", test_configs)
def test_static_cache_tosa_FP(test_data):
    module = StaticCacheModule(test_data).eval()
    pipeline = TosaPipelineFP[input_t](
        module,
        module.get_inputs(),
        aten_op=[],
        exir_op=[],
        transform_passes=[InsertInt32CastsAfterInt64PlaceholdersPass()],
    )
    pipeline.run()


@pytest.mark.xfail(
    reason="TODO(MLETORCH-1818): Quantization for StaticCache is not yet supported."
)
@common.parametrize("test_data", test_configs)
def test_static_cache_tosa_INT(test_data):
    module = StaticCacheModule(test_data).eval()
    pipeline = TosaPipelineINT[input_t](
        module,
        module.get_inputs(),
        aten_op=[],
        exir_op=[],
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@pytest.mark.xfail(
    reason="Quantization for StaticCache is not yet supported. Scatter operator is also not supported on U55."
)
@common.parametrize("test_data", test_configs)
def test_static_cache_u55_INT(test_data):
    module = StaticCacheModule(test_data).eval()
    pipeline = EthosU55PipelineINT[input_t](
        module,
        module.get_inputs(),
        aten_ops=[],
    )
    pipeline.run()


@common.XfailIfNoCorstone320
@pytest.mark.xfail(
    reason="TODO(MLETORCH-1818): Quantization for StaticCache is not yet supported."
)
@common.parametrize("test_data", test_configs)
def test_static_cache_u85_INT(test_data):
    module = StaticCacheModule(test_data).eval()
    pipeline = EthosU85PipelineINT[input_t](module, module.get_inputs(), aten_ops=[])
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("test_data", test_configs)
def test_static_cache_vgf_no_quant(test_data):
    module = StaticCacheModule(test_data).eval()
    pipeline = VgfPipeline[input_t](
        module,
        module.get_inputs(),
        aten_op=[],
        exir_op=[],
        transform_passes=[InsertInt32CastsAfterInt64PlaceholdersPass()],
        quantize=False,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@pytest.mark.xfail(
    reason="TODO(MLETORCH-1818): Quantization for StaticCache is not yet supported."
)
@common.parametrize("test_data", test_configs)
def test_static_cache_vgf_quant(test_data):
    module = StaticCacheModule(test_data).eval()
    pipeline = VgfPipeline[input_t](
        module,
        module.get_inputs(),
        aten_op=[],
        exir_op=[],
        quantize=True,
    )
    pipeline.run()
