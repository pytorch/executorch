# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest

import torch
from executorch.backends.arm._passes import CastInt64BuffersToInt32Pass
from executorch.backends.arm._passes.insert_int32_casts_after_int64_placeholders import (
    InsertInt32CastsAfterInt64PlaceholdersPass,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.arm_tester import RunPasses, ToExecutorch
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
    StaticQuantizedKVCache,
)
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes.init_mutable_pass import InitializedMutableBufferPass
from torch.export.graph_signature import InputKind, OutputKind

from transformers import LlamaConfig
from transformers.cache_utils import StaticCache, StaticLayer

input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


test_configs = {
    "multihead_attention": LlamaConfig(num_attention_heads=32),  # type: ignore[call-arg]
    "grouped_query_attention": LlamaConfig(  # type: ignore[call-arg]
        num_attention_heads=32, num_key_value_heads=4
    ),
    "multi_query_attention": LlamaConfig(num_attention_heads=32, num_key_value_heads=1),  # type: ignore[call-arg]
}


STATIC_CACHE_BUFFER_COUNT = (
    3 if hasattr(StaticLayer(max_cache_len=1), "cumulative_length") else 2
)

EXPECTED_INPUT_COUNTS = {
    InputKind.BUFFER: STATIC_CACHE_BUFFER_COUNT,
    InputKind.USER_INPUT: 3,
}

EXPECTED_OUTPUT_COUNTS = {
    OutputKind.BUFFER_MUTATION: STATIC_CACHE_BUFFER_COUNT,
    OutputKind.USER_OUTPUT: 2,
}

EXPECTED_STATIC_QUANTIZED_INPUT_COUNTS = {
    InputKind.BUFFER: 2,
    InputKind.USER_INPUT: 3,
}

EXPECTED_STATIC_QUANTIZED_OUTPUT_COUNTS = {
    OutputKind.BUFFER_MUTATION: 2,
    OutputKind.USER_OUTPUT: 2,
}


def _initialize_cache_buffers(pipeline, pattern: list[str]) -> None:
    pipeline.change_args(
        "to_executorch",
        ToExecutorch(
            ExecutorchBackendConfig(passes=[InitializedMutableBufferPass(pattern)])
        ),
    )


def _prepare_static_cache_pipeline(pipeline) -> None:
    pipeline.add_stage_after(
        "export",
        pipeline.tester.run_passes,
        RunPasses(passes_with_exported_program=[CastInt64BuffersToInt32Pass]),
    )
    _initialize_cache_buffers(pipeline, ["cache_layer_"])


@torch.no_grad()
class StaticQuantizedCacheModule(torch.nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        max_cache_len: int = 10,
        scale: float = 1.0 / 127.0,
    ) -> None:
        super().__init__()

        self.config = config
        hidden_size = self.config.hidden_size
        num_attention_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads
        assert hidden_size is not None and num_attention_heads is not None
        assert num_key_value_heads is not None

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.cache = StaticQuantizedKVCache(
            max_batch_size=1,
            max_context_length=max_cache_len,
            n_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            scale=scale,
            use_custom_update_cache_op=False,
            use_per_channel=False,
        )

    def forward(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cache.update(cache_position, key_states, value_states)

    def get_inputs(self) -> input_t:
        key_states = torch.randn(
            (
                1,
                self.num_key_value_heads,
                1,
                self.head_dim,
            ),
            dtype=torch.float32,
        )
        value_states = torch.randn(
            (
                1,
                self.num_key_value_heads,
                1,
                self.head_dim,
            ),
            dtype=torch.float32,
        )
        cache_position = torch.tensor([1], dtype=torch.int64)

        return key_states, value_states, cache_position


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
        num_key_value_heads = self.config.num_key_value_heads
        assert hidden_size is not None and num_attention_heads is not None
        assert num_key_value_heads is not None

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.cache.early_initialization(
            1,
            self.num_key_value_heads,
            self.hidden_size // self.num_attention_heads,
            self.dtype,
            torch.device("cpu"),
        )

        for i, layer in enumerate(self.cache.layers):
            self.register_buffer(f"cache_layer_keys_{i}", layer.keys)  # type: ignore[union-attr]
            self.register_buffer(f"cache_layer_values_{i}", layer.values)  # type: ignore[union-attr]
            if hasattr(layer, "cumulative_length") and isinstance(
                layer.cumulative_length, torch.Tensor
            ):
                self.register_buffer(
                    f"cache_layer_cumulative_length_{i}",
                    layer.cumulative_length,
                )

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
                self.num_key_value_heads,
                1,
                self.hidden_size // self.num_attention_heads,
            ),
            dtype=torch.float32,
        )
        value_states = torch.randn(
            (
                1,
                self.num_key_value_heads,
                1,
                self.hidden_size // self.num_attention_heads,
            ),
            dtype=torch.float32,
        )
        cache_position = torch.tensor([1], dtype=torch.int64)

        return key_states, value_states, cache_position


@common.parametrize("test_data", test_configs)
def test_static_cache_export_preserves_updates(test_data):
    max_cache_len = 3
    module = StaticCacheModule(test_data, max_cache_len=max_cache_len).eval()
    exported_module = (
        torch.export.export(module, module.get_inputs(), strict=True)
        .run_decompositions()
        .module()
    )
    expected_key = torch.zeros(
        1,
        test_data.num_key_value_heads,
        max_cache_len,
        test_data.hidden_size // test_data.num_attention_heads,
    )
    expected_value = torch.zeros_like(expected_key)

    for position in range(max_cache_len):
        key_states, value_states, _ = module.get_inputs()
        cache_position = torch.tensor([position], dtype=torch.int64)
        expected_key[:, :, cache_position] = key_states
        expected_value[:, :, cache_position] = value_states

        key, value = exported_module(key_states, value_states, cache_position)

        torch.testing.assert_close(key, expected_key)
        torch.testing.assert_close(value, expected_value)


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
    _prepare_static_cache_pipeline(pipeline)
    pipeline.count_program_io_kinds(EXPECTED_INPUT_COUNTS, EXPECTED_OUTPUT_COUNTS)
    pipeline.run()


@common.parametrize("test_data", test_configs)
def test_static_cache_tosa_INT(test_data):
    module = StaticQuantizedCacheModule(test_data).eval()
    pipeline = TosaPipelineINT[input_t](
        module, module.get_inputs(), aten_op=[], exir_op=[]
    )
    _initialize_cache_buffers(pipeline, ["k_cache", "v_cache"])
    pipeline.count_program_io_kinds(
        EXPECTED_STATIC_QUANTIZED_INPUT_COUNTS, EXPECTED_STATIC_QUANTIZED_OUTPUT_COUNTS
    )
    pipeline.run()


@common.XfailIfNoCorstone300
@pytest.mark.xfail(reason="Scatter operator is not supported on U55.")
@common.parametrize("test_data", test_configs)
def test_static_cache_u55_INT(test_data):
    module = StaticQuantizedCacheModule(test_data).eval()
    pipeline = EthosU55PipelineINT[input_t](
        module,
        module.get_inputs(),
        aten_ops=[],
    )
    _initialize_cache_buffers(pipeline, ["k_cache", "v_cache"])
    pipeline.run()


@common.XfailIfNoCorstone320
@common.parametrize("test_data", test_configs)
def test_static_cache_u85_INT(test_data):
    module = StaticQuantizedCacheModule(test_data).eval()
    pipeline = EthosU85PipelineINT[input_t](
        module,
        module.get_inputs(),
        aten_ops=[],
    )
    _initialize_cache_buffers(pipeline, ["k_cache", "v_cache"])
    # U85: keep _to_dim_order_copy portable for int64->int32 cast of cache_position (not delegatable).
    pipeline.tester.use_portable_ops = True
    pipeline.count_program_io_kinds(
        EXPECTED_STATIC_QUANTIZED_INPUT_COUNTS, EXPECTED_STATIC_QUANTIZED_OUTPUT_COUNTS
    )
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
    _prepare_static_cache_pipeline(pipeline)
    pipeline.count_program_io_kinds(EXPECTED_INPUT_COUNTS, EXPECTED_OUTPUT_COUNTS)
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("test_data", test_configs)
def test_static_cache_vgf_quant(test_data):
    module = StaticQuantizedCacheModule(test_data).eval()
    pipeline = VgfPipeline[input_t](
        module,
        module.get_inputs(),
        aten_op=[],
        exir_op=[],
        quantize=True,
        tosa_spec="TOSA-1.0+INT",
    )
    _initialize_cache_buffers(pipeline, ["k_cache", "v_cache"])
    pipeline.count_program_io_kinds(
        EXPECTED_STATIC_QUANTIZED_INPUT_COUNTS, EXPECTED_STATIC_QUANTIZED_OUTPUT_COUNTS
    )
    pipeline.run()
