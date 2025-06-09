# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Retrieves the pretrained models for Moshi and Mimi."""
from dataclasses import dataclass
from pathlib import Path

import torch.nn as nn

try:
    from huggingface_hub.errors import EntryNotFoundError
except ImportError:
    from huggingface_hub.utils import EntryNotFoundError  # noqa: F401
import typing as tp
from contextlib import ExitStack

import torch
from einops import rearrange
from executorch.examples.qualcomm.oss_scripts.moshi.model.static_convtr import (
    StaticConvTrUpsample1d,
)
from executorch.examples.qualcomm.oss_scripts.moshi.model.static_seanet_decoder import (
    StaticSEANetDecoder,
)

from moshi.models.compression import MimiModel
from moshi.models.loaders import (
    _is_safetensors,
    _quantizer_kwargs,
    _seanet_kwargs,
    _transformer_kwargs,
)
from moshi.modules import SEANetEncoder, transformer
from moshi.modules.resample import ConvDownsample1d
from moshi.modules.rope import RotaryEmbedding
from moshi.modules.streaming import State, StreamingModule
from moshi.modules.transformer import (
    create_norm_fn,
    KVCacheResult,
    LayerScale,
    ProjectedTransformer,
    RingKVCache,
    StreamingTransformer,
    StreamingTransformerLayer,
)
from moshi.quantization import BaseQuantizer, SplitResidualVectorQuantizer
from moshi.utils import quantize
from moshi.utils.compile import no_compile
from safetensors.torch import load_model
from torch.nn import functional as F

SAMPLE_RATE = 24000
FRAME_RATE = 12.5

TEXT_TOKENIZER_NAME = "tokenizer_spm_32k_3.model"
MOSHI_NAME = "model.safetensors"
MOSHI_Q8_NAME = "model.q8.safetensors"
MIMI_NAME = "tokenizer-e351c8d8-checkpoint125.safetensors"
DEFAULT_REPO = "kyutai/moshiko-pytorch-bf16"


class StaticRingKVCache(RingKVCache):
    # Static Mimi Changes:
    # 1) Remove all inplace kv_cache & index updates, perform non inplace updates and return updated output as next execution's input
    # 2) Use end_offset to keep track of nth iteration * 2. Different from end_index, this number does not reset even after end_index resets
    def complete(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        end_index: torch.Tensor,
        end_offset: torch.Tensor,
    ) -> KVCacheResult:
        assert k.shape[:-1] == v.shape[:-1], (k.shape, v.shape)
        B, H, T, D = k.shape

        end_index = torch.where(
            end_index >= self.capacity, end_index - self.capacity, end_index
        )

        assert T > 0

        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        k_cache = k_cache.permute(0, 2, 1, 3)
        v_cache = v_cache.permute(0, 2, 1, 3)
        k_0, k_1 = torch.split(k, 1, dim=1)
        v_0, v_1 = torch.split(v, 1, dim=1)

        index_0 = torch.tensor([0], dtype=torch.int64) + end_index
        index_1 = torch.tensor([1], dtype=torch.int64) + end_index

        k_cache = torch.ops.aten.index_put(k_cache, [None, index_0], k_0)
        k_cache = torch.ops.aten.index_put(k_cache, [None, index_1], k_1)

        v_cache = torch.ops.aten.index_put(v_cache, [None, index_0], v_0)
        v_cache = torch.ops.aten.index_put(v_cache, [None, index_1], v_1)
        k_cache = k_cache.permute(0, 2, 1, 3)
        v_cache = v_cache.permute(0, 2, 1, 3)

        indexes = torch.arange(
            self.capacity, device=end_offset.device, dtype=torch.long
        )

        # end_index correspond to the actual index where the last value was written.
        offset = T - 1
        last_offset = end_offset + offset
        delta = indexes - (end_index + offset)
        # We know that if `index == end_index`, then we should output `self.end_offset`.
        # If `index = end_index - 1` we should output `self.end_offset - 1`
        # If `index = end_index - n` we should output `self.end_offset - n`
        # Now, for `index == end_index + 1` , we actually have the oldest entry in the cache,
        # so we should output `end_index + 1 - self.capacity`
        positions = torch.where(
            delta <= 0,
            last_offset + delta,
            last_offset + delta - self.capacity,
        )
        end_offset = end_offset.add(T)
        end_index = end_index.add(T)
        invalid = indexes >= end_offset
        positions = torch.where(invalid, torch.full_like(positions, -1), positions)
        return (
            KVCacheResult(k_cache, v_cache, positions),
            k_cache,
            v_cache,
            end_index,
            end_offset,
        )


@dataclass
class _StaticMHAState(State):
    kv_cache: StaticRingKVCache
    offset: torch.Tensor
    offset_cpu: int

    def reset(self):
        self.kv_cache.reset()
        self.offset.zero_()
        self.offset_cpu = 0


class StaticStreamingMultiheadAttention(StreamingModule[_StaticMHAState]):
    _fsdp_final = True

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        causal: bool = False,
        context: tp.Optional[int] = None,
        rope: tp.Optional[RotaryEmbedding] = None,
        weights_per_step: int = 0,
        weights_per_step_schedule: list[int] | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.embed_dim = embed_dim
        self.causal = causal
        self.context = context
        self.rope = rope
        self.num_heads = num_heads
        self.weights_per_step = weights_per_step
        self.weights_per_step_schedule = weights_per_step_schedule

        out_dim = embed_dim
        out_dim = 3 * embed_dim
        mult = 1
        in_proj = nn.Linear(embed_dim, mult * out_dim, bias=False, **factory_kwargs)
        # We try to follow the default PyTorch MHA convention, to easily compare results.
        self.in_proj_weight = in_proj.weight
        self.in_proj_bias = in_proj.bias
        self.out_proj = nn.Linear(
            embed_dim, mult * embed_dim, bias=False, **factory_kwargs
        )

    def _init_streaming_state(self, batch_size: int) -> _StaticMHAState:
        capacity = self.context
        device = self.in_proj_weight.device
        dtype = self.in_proj_weight.dtype
        dim_per_head = self.embed_dim // self.num_heads
        self.kv_cache = StaticRingKVCache(
            batch_size, self.num_heads, dim_per_head, capacity, device, dtype
        )
        return _StaticMHAState(
            self.kv_cache,
            offset=torch.zeros(1, device=device, dtype=torch.long),
            offset_cpu=0,
        )

    def _complete_kv(
        self, k, v, k_cache, v_cache, end_index, end_offset
    ) -> KVCacheResult:
        state = self._streaming_state
        # Check here, since we did not override methods used when streaming_state == None
        assert state is not None
        return self.kv_cache.complete(k, v, k_cache, v_cache, end_index, end_offset)

    # Static Mimi Changes:
    # 1) use end_offset to replace state.offset when assigning it to offset variable
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        end_index: torch.Tensor,
        end_offset: torch.Tensor,
    ):
        T = query.shape[1]
        assert self.causal, "Streaming only available for causal"
        offset = end_offset

        if self.weights_per_step:
            projected = quantize.multi_linear(
                self.weights_per_step,
                self.weights_per_step_schedule,
                self,
                query,
                offset_cpu=0,
                name="in_proj_weight",
            )
        else:
            projected = quantize.linear(self, query, "in_proj_weight")
        q, k, v = rearrange(
            projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads
        )

        q, k = self.rope(q, k, offset, time_before_heads=False)
        kv_cache_result, k_cache, v_cache, end_index, end_offset = self._complete_kv(
            k, v, k_cache, v_cache, end_index, end_offset
        )
        k, v, pos_k = kv_cache_result

        pos_k = pos_k.view(1, -1)
        pos_q = offset + torch.arange(T, device=q.device, dtype=torch.long).view(-1, 1)
        delta = pos_q - pos_k
        attn_bias = (pos_k >= 0) & (delta >= 0)
        attn_bias = attn_bias & (delta < self.context)

        x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)

        x = rearrange(x, "b h t d -> b t (h d)")

        x = quantize.linear(self.out_proj, x)

        return x, k_cache, v_cache, end_index, end_offset


class StaticStreamingTransformerLayer(StreamingTransformerLayer):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int | list[int] = 2048,
        causal: bool = False,
        context: tp.Optional[int] = None,
        rope: tp.Optional[RotaryEmbedding] = None,
        norm: str = "layer_norm",
        layer_scale: tp.Optional[float] = None,
        gating: str = "none",
        weights_per_step: int = 0,
        weights_per_step_schedule: list[int] | None = None,
        activation=F.gelu,
        skip_self_attn: bool = False,
        device=None,
        dtype=None,
    ):
        # Skip parent class init and call grandparent directly
        super(StreamingTransformerLayer, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # Redefine self_attn to our streaming multi-head attention
        attn_kwargs: tp.Dict[str, tp.Any] = {
            "embed_dim": d_model,
            "num_heads": num_heads,
        }
        if not skip_self_attn:
            self.self_attn: (
                StaticStreamingMultiheadAttention
            ) = StaticStreamingMultiheadAttention(
                causal=causal,
                context=context,
                rope=rope,
                weights_per_step=weights_per_step,
                weights_per_step_schedule=weights_per_step_schedule,
                **attn_kwargs,  # type: ignore
                **factory_kwargs,  # type: ignore
            )  # type: ignore
            self.norm1 = create_norm_fn(norm, d_model, **factory_kwargs)
        self.norm2 = create_norm_fn(norm, d_model, **factory_kwargs)
        # Redefine feedforward layers to expose bias parameter
        self.weights_per_step = weights_per_step
        self.weights_per_step_schedule = weights_per_step_schedule
        self.gating: tp.Optional[nn.Module] = None
        self.linear1: tp.Optional[nn.Module] = None
        self.linear2: tp.Optional[nn.Module] = None
        self.activation = activation
        self.skip_self_attn = skip_self_attn

        assert (
            not weights_per_step
        ), "weights_per_step without gating not supported for now."
        assert not isinstance(
            dim_feedforward, list
        ), "List dim_feedforward without gating not supported for now."
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False, **factory_kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False, **factory_kwargs)

        self.layer_scale_1 = LayerScale(d_model, layer_scale, **factory_kwargs)  # type: ignore
        self.layer_scale_2 = LayerScale(d_model, layer_scale, **factory_kwargs)  # type: ignore

    def _sa_block(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        end_index: torch.Tensor,
        end_offset: torch.Tensor,
    ):
        x_orig = x
        x = self.norm1(x)
        update, k_cache, v_cache, end_index, end_offset = self.self_attn(
            x, x, x, k_cache, v_cache, end_index, end_offset
        )
        x = x_orig.to(update) + self.layer_scale_1(update)
        return x, k_cache, v_cache, end_index, end_offset

    def forward(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        end_index: torch.Tensor,
        end_offset: torch.Tensor,
    ):
        with ExitStack() as stack:
            if x.device.type != "cuda":
                stack.enter_context(no_compile())
            x, k_cache, v_cache, end_index, end_offset = self._sa_block(
                x, k_cache, v_cache, end_index, end_offset
            )
            x = self._ff_block(x)
            return x, k_cache, v_cache, end_index, end_offset


class StaticStreamingTransformer(StreamingTransformer):
    # Static Mimi Changes:
    # 1) After static variables are passed in, unbind them and pass them to corresponding transformer layers.
    #    After function returned, stack all layers back to 1 tensor and return it.
    # 2) Remove other positional embeddings logic such as "sin", "sine_rope" since "rope" is used.
    def forward(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        end_index: torch.Tensor,
        end_offset: torch.Tensor,
        *args,
        **kwargs,
    ):
        B, T, C = x.shape

        dtype_input = x.dtype
        k_cache_list = torch.unbind(k_cache, dim=0)
        v_cache_list = torch.unbind(v_cache, dim=0)
        end_index_list = torch.unbind(end_index, dim=0)
        end_offset_list = torch.unbind(end_offset, dim=0)
        new_k_cache = []
        new_v_cache = []
        new_end_index = []
        new_end_offset = []
        for i, layer in enumerate(self.layers):
            x, k_cache_res, v_cache_res, end_index_res, end_offset_res = layer(
                x,
                k_cache_list[i],
                v_cache_list[i],
                end_index_list[i],
                end_offset_list[i],
                *args,
                **kwargs,
            )
            new_k_cache.append(k_cache_res)
            new_v_cache.append(v_cache_res)
            new_end_index.append(end_index_res)
            new_end_offset.append(end_offset_res)
        new_k_cache = torch.stack(new_k_cache, dim=0)
        new_v_cache = torch.stack(new_v_cache, dim=0)
        new_end_index = torch.stack(new_end_index, dim=0)
        new_end_offset = torch.stack(new_end_offset, dim=0)
        return (
            x.to(dtype_input),
            new_k_cache,
            new_v_cache,
            new_end_index,
            new_end_offset,
        )


class StaticProjectedTransformer(ProjectedTransformer):
    """Transformer with optional projections of the input and output to different dimensions when needed.
    Supports multiple outputs.

    Args:
        input_dimension (int): dimension of the input.
        output_dimensions (tuple[int]): dimensions of the outputs.
        d_model (int): inner dimension of the Transformer.
        conv_layout (bool): If True, expects `[B, C, T]` shaped tensors, otherwise, `[B, T, C]`.
            Similarly, the output will have the same layout.
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimensions: tp.Tuple[int, ...],
        d_model: int,
        *,
        conv_layout: bool = False,
        **kwargs,
    ):
        super(ProjectedTransformer, self).__init__()
        self.transformer = StaticStreamingTransformer(d_model=d_model, **kwargs)
        self.input_dimension = input_dimension
        self.output_dimensions = output_dimensions
        self.conv_layout = conv_layout
        self.input_proj = None
        if d_model != input_dimension:
            self.input_proj = nn.Linear(input_dimension, d_model, bias=False)

        self.output_projs = nn.ModuleList()
        for output_dimension in output_dimensions:
            if d_model == output_dimension:
                self.output_projs.append(nn.Identity())
            else:
                self.output_projs.append(
                    nn.Linear(d_model, output_dimension, bias=False)
                )

    def forward(
        self,
        x,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        end_index: torch.Tensor,
        end_offset: torch.Tensor,
        *args,
        **kwargs,
    ):
        if self.conv_layout:
            x = x.transpose(1, 2)
        if self.input_proj is not None:
            x = self.input_proj(x)
        z, k_cache, v_cache, end_index, end_offset = self.transformer(
            x, k_cache, v_cache, end_index, end_offset, *args, **kwargs
        )
        ys = []
        for output_proj in self.output_projs:
            y = output_proj(z)
            if self.conv_layout:
                y = y.transpose(1, 2)
            ys.append(y)
        return ys, k_cache, v_cache, end_index, end_offset


class StaticMimiModel(MimiModel):
    """
    Static Mimi Model does not keep track of any states inside the model.
    It moves all the state related variables to I/O since lowered model does not support keep track of the states in the model.
    Static variables includes indices, offsets, kv cache, partials, and previous.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: BaseQuantizer,
        frame_rate: float,
        encoder_frame_rate: float,
        sample_rate: int,
        channels: int,
        causal: bool = False,
        encoder_transformer: tp.Optional[nn.Module] = None,
        decoder_transformer: tp.Optional[nn.Module] = None,
        resample_method: str = "interpolate",
        upsample_channel_wise_bug: bool = True,
        freeze_encoder: bool = False,
        freeze_quantizer: bool = False,
        freeze_quantizer_level: int = -1,
        torch_compile_encoder_decoder: bool = False,
    ):
        super(MimiModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_transformer = encoder_transformer
        self.decoder_transformer = decoder_transformer
        self.quantizer = quantizer
        self._frame_rate = frame_rate
        self._sample_rate = sample_rate
        self._channels = channels
        self.encoder_frame_rate = encoder_frame_rate
        self.torch_compile_encoder_decoder = torch_compile_encoder_decoder
        self.freeze_quantizer = freeze_quantizer
        self.freeze_quantizer_level = (
            freeze_quantizer_level
            if freeze_quantizer_level > 0
            else self.quantizer.num_codebooks
        )

        # We will need the dimension for the resampling. In general the encoder will be a SeanetEncoder
        # which exposes a `dimension` attribute.
        dimension = encoder.dimension
        assert isinstance(
            dimension, int
        ), f"Dimension should be int, got {dimension} of type {type(dimension)}."
        self.dimension = dimension

        assert resample_method in [
            "interpolate",
            "conv",
            "avg_pool",
        ], f"Invalid resample_method {resample_method}"
        self.resample_method = resample_method
        if encoder_frame_rate != frame_rate:
            assert not (
                causal and resample_method == "interpolate"
            ), "Cannot interpolate with causal model."
            if resample_method in ["conv", "avg_pool"]:
                assert (
                    self.encoder_frame_rate > self.frame_rate
                ), "Cannot upsample with conv."
                downsample_stride = self.encoder_frame_rate / self.frame_rate
                assert downsample_stride == int(
                    downsample_stride
                ), f"Only integer strides are supported, got {downsample_stride}"
                learnt = resample_method == "conv"
                self.downsample = ConvDownsample1d(
                    int(downsample_stride),
                    dimension=dimension,
                    learnt=learnt,
                    causal=causal,
                )
                self.upsample = StaticConvTrUpsample1d(
                    int(downsample_stride),
                    dimension=dimension,
                    learnt=learnt,
                    causal=causal,
                    channel_wise=upsample_channel_wise_bug,
                )

    def _static_to_encoder_framerate(self, x: torch.Tensor, partial: torch.Tensor):
        # Convert from overall framerate to the encoder frame rate.
        x, partial = self.upsample(x, partial)
        return x, partial

    def decode(
        self,
        codes,
        k_cache,
        v_cache,
        end_index,
        end_offset,
        partial_convtr_0,
        partial_convtr_1,
        partial_convtr_2,
        partial_convtr_3,
        partial_convtr_4,
        previous_conv_0,
        previous_conv_1,
        previous_conv_3,
        previous_conv_5,
        previous_conv_7,
        previous_conv_9,
    ):
        state = self._streaming_state
        emb = self.decode_latent(codes)
        emb, partial_convtr_0 = self._static_to_encoder_framerate(emb, partial_convtr_0)
        assert state is not None
        (emb,), k_cache, v_cache, end_index, end_offset = self.decoder_transformer(
            emb, k_cache, v_cache, end_index, end_offset
        )
        with self._context_for_encoder_decoder:
            (
                out,
                partial_convtr_1,
                partial_convtr_2,
                partial_convtr_3,
                partial_convtr_4,
                previous_conv_0,
                previous_conv_1,
                previous_conv_3,
                previous_conv_5,
                previous_conv_7,
                previous_conv_9,
            ) = self.decoder(
                emb,
                partial_convtr_1,
                partial_convtr_2,
                partial_convtr_3,
                partial_convtr_4,
                previous_conv_0,
                previous_conv_1,
                previous_conv_3,
                previous_conv_5,
                previous_conv_7,
                previous_conv_9,
            )
        return (
            out,
            k_cache,
            v_cache,
            end_index,
            end_offset,
            partial_convtr_0,
            partial_convtr_1,
            partial_convtr_2,
            partial_convtr_3,
            partial_convtr_4,
            previous_conv_0,
            previous_conv_1,
            previous_conv_3,
            previous_conv_5,
            previous_conv_7,
            previous_conv_9,
        )


class StaticMimiDecoderModel(StaticMimiModel):
    def forward(self, *args):
        return super().decode(*args)


def get_static_mimi(
    filename: str | Path, device: torch.device | str = "cpu", num_codebooks: int = 8
) -> MimiModel:
    """Return a pretrained Mimi model."""
    encoder = SEANetEncoder(**_seanet_kwargs)
    decoder = StaticSEANetDecoder(**_seanet_kwargs)
    _transformer_kwargs["layer_class"] = StaticStreamingTransformerLayer
    encoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    decoder_transformer = StaticProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    quantizer = SplitResidualVectorQuantizer(
        **_quantizer_kwargs,
    )
    model = StaticMimiDecoderModel(
        encoder,
        decoder,
        quantizer,
        channels=1,
        sample_rate=SAMPLE_RATE,
        frame_rate=FRAME_RATE,
        encoder_frame_rate=SAMPLE_RATE / encoder.hop_length,
        causal=True,
        resample_method="conv",
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
    ).to(device=device)
    model.eval()
    if _is_safetensors(filename):
        load_model(model, filename, strict=True)
    else:
        pkg = torch.load(filename, "cpu")  # noqa: TOR102
        model.load_state_dict(pkg["model"])
    model.set_num_codebooks(num_codebooks)
    return model
