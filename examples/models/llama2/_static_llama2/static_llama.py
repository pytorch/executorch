# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path
from typing import List, Optional, Tuple

import pkg_resources

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from executorch import exir
from executorch.examples.models.llama2.fairseq2 import convert_to_llama_checkpoint
from executorch.examples.models.llama2.llama_transformer import ModelArgs


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        config,
        max_seq_len: int = 128,
        base: int = 10000,
        device=None,
    ):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, config.dim, 2).float().to(device) / config.dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_seq_len
        t = torch.arange(
            self.max_seq_len_cached,
            device="cpu",
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, n_heads, seq_len, head_size]
        print(
            "LlamaRotaryEmbedding::forward: ",
            " x.shape: ",
            x.shape,
            " seq_len: ",
            seq_len,
        )

        return (
            torch.ops.aten.slice(self.cos_cached, 2, 0, seq_len).to(dtype=x.dtype),
            torch.ops.aten.slice(self.sin_cached, 2, 0, seq_len).to(dtype=x.dtype),
        )

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.n_heads
        self.head_dim = self.dim // self.num_heads
        self.n_kv_heads = config.n_kv_heads
        self.num_key_value_groups = self.num_heads // self.n_kv_heads
        # self.max_seq_len = config.max_seq_len

        if self.head_dim % 2 != 0:
            raise ValueError(f"Head dim must be even. Got {self.head_dim}.")

        if (self.head_dim * self.num_heads) != self.dim:
            raise ValueError(
                f"dim must be divisible by num_heads (got `dim`: {self.dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.wq = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.dim, bias=False)

        self.attn_softmax = torch.nn.Softmax(dim=-1)

        scale = float(self.head_dim) ** -0.5
        scale_tensor = torch.tensor(
            [scale], dtype=torch.float32, requires_grad=False
        ).view(1, 1, 1)
        self.register_buffer("scale_tensor", scale_tensor, False)

    def rotate_half_q(self, x: torch.Tensor):
        """Rotates half the hidden dims of the input."""
        # x1 = x[..., : self.head_dim // 2]
        # x2 = x[..., self.head_dim // 2 :]
        # print(f"        apply_rotary_emb::rotate_half_q::x: {x.shape}", x)
        x1, x2 = x.float().reshape(x.shape[:-1] + (-1, 2)).unbind(-1)
        # print(f"        apply_rotary_emb::rotate_half_q::x1: {x1.shape}", x1)
        # print(f"        apply_rotary_emb::rotate_half_q::x2: {x2.shape}", x2)
        return torch.cat((-1 * x2, x1), dim=-1)

    def rotate_half_k(self, x: torch.Tensor):
        """Rotates half the hidden dims of the input."""
        # print(f"        apply_rotary_emb::rotate_half_k::x: {x.shape}", x)
        # print(f"        apply_rotary_emb::rotate_half_k::x.shape[:1]: ", x.shape[:1])
        # print(f"        apply_rotary_emb::rotate_half_k::x.shape[2:]: ", x.shape[2:])
        # print(f"        apply_rotary_emb::rotate_half_k::x.shape[:1] + (-1, 2) + x.shape[2:]: ", x.shape[:1] + (-1, 2) + x.shape[2:])
        # x1 = x[..., : self.head_dim // 2, :]
        # x2 = x[..., self.head_dim // 2 :, :]
        # x1, x2 = x.float().reshape(x.shape[:1] + (-1, 2) + x.shape[2:]).unbind(-1)
        x1, x2 = x.float().reshape(x.shape[:1] + (-1, 2) + x.shape[2:]).unbind(-2)
        # print(f"        apply_rotary_emb::rotate_half_k::x1: {x1.shape}", x1)
        # print(f"        apply_rotary_emb::rotate_half_k::x2: {x2.shape}", x2)
        return torch.cat((-1 * x2, x1), dim=-2)

    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        cos_transpose: torch.Tensor,
        sin: torch.Tensor,
        sin_transpose: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rotated_q = self.rotate_half_q(q)
        print(f"        apply_rotary_emb::cos: {cos.shape}", cos)
        print(f"        apply_rotary_emb::sin: {sin.shape}", sin)
        print(f"        apply_rotary_emb::q: {q.shape}", q)
        print(f"        apply_rotary_emb::rotated_q: {rotated_q.shape}", rotated_q)
        q_embed = q * cos + rotated_q * sin
        # print(f"        apply_rotary_emb::q_embed: {q_embed.shape}", q_embed)
        # print("k.shape: ", k.shape, "cos_transpose.shape: ", cos_transpose.shape)

        rotated_k = self.rotate_half_k(k)
        # print(f"        apply_rotary_emb::k: {k.shape}", k)
        # print(f"        apply_rotary_emb::rotated_k: {rotated_k.shape}", rotated_k)
        # print(f"        apply_rotary_emb::cos_transpose: {cos_transpose.shape}", cos_transpose)

        k_embed = (k * cos_transpose) + (rotated_k * sin_transpose)
        # print(f"        apply_rotary_emb::k_embed: {k_embed.shape}", k_embed)

        return q_embed, k_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        cos_transpose: torch.Tensor,
        sin: torch.Tensor,
        sin_transpose: torch.Tensor,
        attention_mask: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_mask: torch.Tensor,
        v_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        S: sequence length
        N: batch size
        head_dim: embed_dim / n_heads
        """
        # print("    Attention::hidden_states: ", )
        print(f"    Attention::hidden_states ({hidden_states.shape}): ", hidden_states)
        query = hidden_states.transpose(0, 1)
        key = query
        value = query

        L, N, embed_dim = query.shape
        print("query: L, ", L, " N: ", N, " embed_dim: ", embed_dim)
        S = key.shape[0]
        # shape (S, N, embed_dim)
        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)

        # shape (N * num_heads, S, head_dim)
        q = q.view(L, N * self.num_heads, self.head_dim).transpose(0, 1)
        # shape (N * num_heads, head_dim, S)
        k = k.view(S, N * self.num_heads, self.head_dim).permute(1, 2, 0)
        # shape (N * num_heads, S, head_dim)
        v = v.view(S, N * self.num_heads, self.head_dim).transpose(0, 1)

        # (ifedorov): cos / sin need to be calculated properly in the runtime based on where we are in the sequence
        cos_L = cos[:L]
        # print("    cos_L.shape: ", cos_L.shape, " L:", L)
        # print(f"    Attention::q ({q.shape}): ", q)
        # print(f"    Attention::cos ({cos.shape}): ", cos)
        # print(f"    Attention::q before apply_rotary_emb ({q.shape}): ", q)
        # print(f"    Attention::k] before apply_rotary_emb ({k.shape}): ", k)
        # print(f"    Attention::cos] before apply_rotary_emb ({cos.shape}): ", cos)
        q, k = self.apply_rotary_pos_emb(
            q,
            k,
            cos,
            cos_transpose,
            sin,
            sin_transpose,
        )
        print(f"    Attention::q after apply rotary_pos ({q.shape}): ", q)
        print(f"    Attention::k after apply rotary_pos ({k.shape}): ", k)

        # Use the incoming cache
        if self.config.use_cache:
            print(f"    Attention::k ({k.shape}): ", k)
            print(f"    Attention::k_cache ({k_cache.shape}): ", k_cache)
            print(f"    Attention::k_mask ({k_mask.shape}): ", k_mask)
            k = (
                k_cache * (1.0 - k_mask) + k * k_mask
            )  # k_mask: [l_max, num_dim] all 0s execpt point of position
            print(f"    Attention::v ({v.shape}): ", v)
            print(f"    Attention::v_cache ({v_cache.shape}): ", v_cache)
            print(f"    Attention::v_mask ({v_mask.shape}): ", v_mask)
            v = v_cache * (1.0 - v_mask) + v * v_mask

        # shape (N * num_heads, L, S)

        # q = q * self.scale_tensor
        print("    Attention::self.scale_tensor: ", self.scale_tensor)

        # shape (N * num_heads, S, max_sequence_length)
        attn = q @ k
        # attn = torch.matmul(q, k)
        print(f"    Attention::qk: {attn.shape}", attn)
        attn = attn * self.scale_tensor

        # print("    Attention::attn: ", attn.shape)

        # import pdb; pdb.set_trace()

        # Attention mask
        print("    Attention::self.scale_tensor: ", self.scale_tensor)
        print(f"    Attention::attn: {attn.shape}", attn)
        print(f"    Attention::attention_mask: {attention_mask.shape}", attention_mask)
        attn += attention_mask

        # softmax
        attn = self.attn_softmax(attn)

        # shape (N * num_heads, L, head_dim)
        y = torch.matmul(attn, v)
        # shape (N, L, embed_dim)
        y = y.transpose(0, 1).contiguous().view(N, L, embed_dim)

        # shape (N, L, num_heads * head_dim)
        y = self.wo(y)
        return y, k, v


class LlamaMLP(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
    ):
        super().__init__()
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        hidden_dim = config.hidden_dim
        if hidden_dim is None:
            # If hidden_dim is not explicitly set in the ModelArgs,
            # then calculate implicitly based on dim and also multiple of `args.multiple_of`
            multiple_of = config.multiple_of
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dim = config.dim
        self.attention = LlamaAttention(config=config)
        self.feed_forward = LlamaMLP(
            config
            # dim=self.dim,
            # intermediate_size=config.intermediate_size,
            # hidden_act=config.hidden_act,
        )
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        # self.input_layernorm = norm_builder(
        #     # layer_norm_type=config.norm_type,
        #     dim=config.dim,
        #     eps=config.norm_eps,
        # )
        # self.post_attention_layernorm = norm_builder(
        #     # layer_norm_type=config.norm_type,
        #     dim=config.dim,
        #     eps=config.norm_eps,
        # )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        cos_transpose: torch.Tensor,
        sin: torch.Tensor,
        sin_transpose: torch.Tensor,
        attention_mask: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_mask: torch.Tensor,
        v_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor]:

        residual = hidden_states.clone()

        hidden_states = self.attention_norm(hidden_states)

        # Self Attention
        hidden_states, k_cache, v_cache = self.attention(
            hidden_states=hidden_states,
            cos=cos,
            cos_transpose=cos_transpose,
            sin=sin,
            sin_transpose=sin_transpose,
            attention_mask=attention_mask,
            k_cache=k_cache,
            v_cache=v_cache,
            k_mask=k_mask,
            v_mask=v_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, k_cache, v_cache


class LlamaModel(nn.Module):
    """
    Transformer decoder consisting of *config.n_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        # self.remove_dynamism: bool = config.remove_dynamism
        print("config: ", config)
        self.use_cache: bool = config.use_kv_cache
        self.padding_idx = None
        self.vocab_size = config.vocab_size
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_embeddings = nn.Embedding(
            config.vocab_size, config.dim, self.padding_idx
        )

        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        tokens: torch.Tensor,
        cos: torch.Tensor,
        cos_transpose: torch.Tensor,
        sin: torch.Tensor,
        sin_transpose: torch.Tensor,
        attention_mask: torch.Tensor,
        k_cache: List[torch.Tensor],
        v_cache: List[torch.Tensor],
        k_mask: List[torch.Tensor],
        v_mask: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        print("tokens: ", tokens)
        inputs_embeds = self.tok_embeddings(tokens)

        batch_size, seq_length, _ = inputs_embeds.shape
        hidden_states = inputs_embeds
        output_k_cache = []
        output_v_cache = []

        for ind in range(self.config.n_layers):
            print("layer index: ", ind)
            k = k_cache[ind]
            v = v_cache[ind]
            km = k_mask[ind]
            vm = v_mask[ind]
            decoder_layer = self.layers[ind]
            print("  hidden_states: ", hidden_states)

            layer_outputs, k, v = decoder_layer(
                hidden_states,
                cos=cos,
                cos_transpose=cos_transpose,
                sin=sin,
                sin_transpose=sin_transpose,
                attention_mask=attention_mask,
                k_cache=k,
                v_cache=v,
                k_mask=km,
                v_mask=vm,
            )
            output_k_cache.append(k)
            output_v_cache.append(v)
            hidden_states = layer_outputs
            print("  updated hidden_states: ", hidden_states)

        hidden_states = self.norm(hidden_states)
        print("  Last normalize hidden_states: ", hidden_states)

        logits = self.output(hidden_states)

        print("  logits: ", logits)

        return logits, k_cache, v_cache


class LlamaForCausalLM(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # default path to the resource file
        # It currently supports 3 ways of specifying the checkpoint location:
        # 1. Using default path locates in examples/models/llama2/params
        # 2. Passing in the checkpoint path and params via kwargs
        # 3. Using the path from pkg_resources, only works with buck2
        try:
            # The 3rd way, if we can import this path, we are running with buck2, all resources can be accessed with pkg_resources.resource_filename
            # pyre-ignore
            from executorch.examples.models.llama2 import params

            ckpt_dir = Path(
                pkg_resources.resource_filename(
                    "executorch.examples.models.llama2", "params"
                )
            )
        except:
            # The 1st way
            ckpt_dir = Path(__file__).absolute().parent / "params"

        checkpoint_path = (
            kwargs["checkpoint"]
            if "checkpoint" in kwargs
            else ckpt_dir / "demo_rand_params.pth"
        )

        params_path = (
            kwargs["params"] if "params" in kwargs else ckpt_dir / "demo_config.json"
        )

        checkpoint_path = (
            kwargs["checkpoint"]
            if "checkpoint" in kwargs
            else ckpt_dir / "demo_rand_params.pth"
        )

        params_path = (
            kwargs["params"] if "params" in kwargs else ckpt_dir / "demo_config.json"
        )
        with open(params_path, "r") as f:
            params = json.loads(f.read())
        max_seq_len = 128
        max_batch_size = 1
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            use_kv_cache=False,
            **params,
        )
        self.config = model_args
        with torch.device("meta"):
            self.model = LlamaModel(model_args)
        print("checkpoint_path: ", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)
        fairseq2_checkpoint = True
        if fairseq2_checkpoint:
            print("Using fairseq2 checkpoint")
            checkpoint = convert_to_llama_checkpoint(checkpoint=checkpoint)

    def get_all_inputs(self, input_ids: torch.LongTensor):
        dim = self.config.dim
        prefill = True
        seq_len = input_ids.shape[1]
        # max_seq_len = self.config.max_sequence_length
        n_heads = self.config.n_heads
        head_dim = self.config.dim // self.config.n_heads
        print("head_dim: ", head_dim)

        # 1st way
        rotary_emb = LlamaRotaryEmbedding(head_dim, self.config)
        cos = torch.zeros(seq_len if prefill else 1, 1 * n_heads, head_dim).transpose(
            0, 1
        )  # [number_attention_heads, seq_len, head_dim]
        sin = torch.zeros(seq_len if prefill else 1, 1 * n_heads, head_dim).transpose(
            0, 1
        )
        print("cos.shape: ", cos.shape)

        cos, sin = rotary_emb(
            cos,  # [number_attention_heads, seq_len, head_dim]
            seq_len=seq_len,
        )
        cos = cos.squeeze(1).squeeze(0)
        sin = sin.squeeze(1).squeeze(0)
        print("input_ids", input_ids)
        # cos = cos[inputs_embeds].unsqueeze(1)
        # sin = sin[inputs_embeds].unsqueeze(1)
        cos = torch.squeeze(cos, 1).expand([n_heads, -1, -1])
        sin = torch.squeeze(sin, 1).expand([n_heads, -1, -1])
        # cos_transpose = cos.transpose(1, 2)
        # sin_transpose = sin.transpose(1, 2)
        cos_transpose = cos.permute(0, 2, 1)
        sin_transpose = sin.permute(0, 2, 1)

        attention_mask = torch.full(
            (
                1 * self.config.n_heads,
                seq_len if prefill else 1,
                seq_len if prefill else seq_len + 1,
            ),
            float("-inf"),
        )
        attention_mask = torch.triu(attention_mask, diagonal=1)

        # k_mask = torch.zeros_like(cos)
        # v_mask = torch.zeros_like(sin)
        # k_mask = [torch.zeros(config.max_batch_size, head_dim, config.max_sequence_length)] * config.n_layers
        # v_mask = [torch.zeros(config.max_batch_size, head_dim, config.max_sequence_length)] * config.n_layers
        k_mask = [
            torch.zeros(self.config.max_batch_size, head_dim, seq_len)
        ] * self.config.n_layers
        v_mask = [
            torch.zeros(self.config.max_batch_size, seq_len, head_dim)
        ] * self.config.n_layers

        # print("k_mask: ", k_mask.shape)

        # print("k_mask: ", k_mask.shape)

        # k_cache: List[torch.Tensor] = [torch.randn(5, 6, 51)] * config.n_layers
        # v_cache: List[torch.Tensor] = [torch.randn(5, 51, 6)] * config.n_layers
        # k_cache: List[torch.Tensor] = [torch.randn(config.max_batch_size, head_dim, config.max_sequence_length)] * config.n_layers
        # v_cache: List[torch.Tensor] = [torch.randn(config.max_batch_size, config.max_sequence_length, head_dim)] * config.n_layers
        k_cache: List[torch.Tensor] = [
            torch.randn(self.config.max_batch_size, head_dim, seq_len)
        ] * self.config.n_layers
        v_cache: List[torch.Tensor] = [
            torch.randn(self.config.max_batch_size, seq_len, head_dim)
        ] * self.config.n_layers
        return {
            "input_ids": input_ids,
            "cos": cos,
            "cos_transpose": cos_transpose,
            "sin": sin,
            "sin_transpose": sin_transpose,
            "attention_mask": attention_mask,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "k_mask": k_mask,
            "v_mask": v_mask,
        }

    def forward(
        self,
        input_ids: torch.LongTensor,
        cos: torch.Tensor,
        cos_transpose: torch.Tensor,
        sin: torch.Tensor,
        sin_transpose: torch.Tensor,
        attention_mask: torch.Tensor,
        k_cache: List[torch.Tensor],
        v_cache: List[torch.Tensor],
        k_mask: List[torch.Tensor],
        v_mask: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        print("input_ids: ", input_ids.shape, input_ids)
        print("cos: ", cos.shape)
        print("cos_transpose: ", cos_transpose.shape)
        print("sin: ", sin.shape)
        print("sin_transpose: ", sin_transpose.shape)
        print("attention_mask: ", attention_mask.shape)
        print("k_cache: ", len(k_cache))
        print("v_cache: ", len(v_cache))
        print("k_mask: ", len(k_mask), k_mask[0].shape)
        print("v_mask: ", len(v_mask), v_mask[0].shape)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        logits, k_cache, v_cache = self.model(
            input_ids,
            cos,
            cos_transpose,
            sin,
            sin_transpose,
            attention_mask,
            k_cache,
            v_cache,
            k_mask,
            v_mask,
        )

        return logits, k_cache, v_cache
