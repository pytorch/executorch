# @lint-ignore-every LICENSELINT
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

# Please refer to README.md in the same folder for more information.

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from executorch.examples.models.llama.attention import (
    Attention,
    ATTENTION_REGISTRY,
    ForwardOptions,
)
from executorch.examples.models.llama.feed_forward import FeedForward
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.norm import RMSNorm
from executorch.examples.models.llama.rope import Rope
from torch import nn


class ConditionalFeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        hidden_dim = args.hidden_dim
        if hidden_dim is None:
            # If hidden_dim is not explicitly set in the ModelArgs,
            # then calculate implicitly based on dim and also multiple of `args.multiple_of`
            multiple_of = args.multiple_of
            hidden_dim = 4 * self.dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Parameter(torch.randn(args.num_experts, hidden_dim, self.dim))
        self.w2 = nn.Parameter(torch.randn(args.num_experts, hidden_dim, self.dim))
        self.w3 = nn.Parameter(torch.randn(args.num_experts, hidden_dim, self.dim))
        self.num_experts = args.num_experts

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        w1_weights = self.w1[expert_indices].transpose(-1, -2)  # [T, A, D, D]
        w3_weights = self.w3[expert_indices].transpose(-1, -2)  # [T, A, D, D]
        w2_weights = self.w2[expert_indices]  # [T, A, D, D]
        x1 = F.silu(torch.einsum("ti,taio -> tao", x, w1_weights))
        x3 = torch.einsum("ti, taio -> tao", x, w3_weights)
        expert_outs = torch.einsum("tao, taoi -> tai", (x1 * x3), w2_weights)
        return expert_outs


class MOEFeedForward(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.dim, config.num_experts, bias=False)
        self.cond_ffn = ConditionalFeedForward(config)
        self.dim = config.dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.dim)
        # T = num_tokens, E = num_experts, D = hidden dim, A = activated experts
        # x: [T, D]
        scores = self.gate(x)  # [T, E]
        expert_weights, expert_indices = torch.topk(scores, 2, dim=-1)  # [T, A], [T, A]
        expert_weights = expert_weights.softmax(dim=-1)  # [T, A]
        expert_outs = self.cond_ffn(x, expert_indices)
        return torch.einsum("tai,ta -> ti", expert_outs, expert_weights)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, attention: Attention):
        """
        Transformer block with support for pre-norm and post-norm.
        Args:
            args (ModelArgs): model configuration parameters.
            attention (Attention): attention object to use in the transformer
                block. See `attention.py` for types of attention. Make sure
                the attention type is registered in the ATTENTION_REGISTRY.
        """
        super().__init__()
        self.use_kv_cache = args.use_kv_cache
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.head_dim
        self.attention = attention

        assert (
            args.hidden_dim is not None
        ), "`hidden_dim` must be set in ModelArgs to construct a TransformerBlock."
        if args.moe:
            self.block_sparse_moe = MOEFeedForward(args)
        else:
            self.feed_forward = FeedForward(dim=args.dim, hidden_dim=args.hidden_dim)

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    @classmethod
    def from_type(cls, layer_id, args, rope) -> "TransformerBlock":
        """
        Create a TransformerBlock with the legacy constructor.
        Args:
            layer_id (int): the index of the layer.
            args (ModelArgs): model configuration parameters.
            rope (Rope): the rope object to use for rotary embeddings.
        """
        if args.attention_type not in ATTENTION_REGISTRY:
            raise ValueError(
                f"Unknown attention type: {args.attention_type}. "
                f"Available: {list(ATTENTION_REGISTRY.keys())}"
            )
        cls = ATTENTION_REGISTRY[args.attention_type]
        attention = cls(args, layer_id, rope, **args.attention_kwargs)
        return TransformerBlock(args, attention)

    def forward(self, x, freqs_cos, freqs_sin, attn_options: ForwardOptions):  # x: 1xN
        h, attn_options_update = self.attention.forward(
            self.attention_norm(x), freqs_cos, freqs_sin, **attn_options
        )

        h = x + h
        if hasattr(self, "block_sparse_moe"):
            out = h + self.block_sparse_moe(self.ffn_norm(h))
        else:
            out = h + self.feed_forward(self.ffn_norm(h))
        return out, attn_options_update


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, layers: nn.ModuleList, rope: Rope):
        """
        Transformer model.
        Args:
            params (ModelArgs): model configuration parameters.
            layers (nn.ModuleList): list of transformer blocks - see the
                `TransformerBlock` type above.
            rope (Rope): the rope object to use for rotary embeddings.
        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.apply_embedding = params.apply_embedding
        self.apply_output = params.apply_output

        self.tok_embeddings = (
            nn.Embedding(params.vocab_size, params.dim)
            if self.apply_embedding
            else None
        )
        self.layers = layers
        self.rope = rope
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = (
            nn.Linear(params.dim, params.vocab_size, bias=False)
            if self.apply_output
            else None
        )
        self.use_kv_cache = params.use_kv_cache
        self.generate_full_logits = params.generate_full_logits
        self.max_seq_len = params.max_seq_len
        self.max_context_len = params.max_context_len
        self.input_prune_map = params.input_prune_map
        self.output_prune_map = params.output_prune_map

    def forward(
        self,
        tokens: Optional[torch.LongTensor] = None,  # tokens
        attn_options: Optional[ForwardOptions] = None,
        h: Optional[torch.FloatTensor] = None,  # embeddings
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Any]]]:
        if (tokens is None) ^ (h is not None):
            raise ValueError(
                "You cannot specify both tokens and h at the same time, and must specify either one"
            )
        if self.apply_embedding and tokens is not None and h is None:
            h = self.tok_embeddings(tokens)

        if attn_options is None:
            attn_options = {}
        seqlen = h.shape[1]
        freqs_cos, freqs_sin = self.rope.get_freqs(
            attn_options.get("input_pos"), seqlen
        )

        # Make a shallow copy so the updates don't get captured by export
        attn_options_ = attn_options.copy() if attn_options is not None else {}
        attn_options_update = None
        for layer in self.layers:
            h, attn_options_update = layer(h, freqs_cos, freqs_sin, attn_options_)
            if attn_options_update is not None:
                attn_options_.update(**attn_options_update)

        if not self.generate_full_logits:
            # Only the last logit is used for the new generated token
            pos = attn_options.get("last_valid_token_pos", -1)
            h = h[:, pos, :]

        h = self.norm(h)

        if self.apply_output:
            logits = self.output(h)

            if self.output_prune_map is not None:
                # expand to original size so that downstream applications can use the logits as-is.
                if self.generate_full_logits:
                    # (1, seq_len, pruned_size) -> (1, seq_len, original_size)
                    expanded_logits = torch.full(
                        [logits.shape[0], logits.shape[1], self.vocab_size],
                        float("-inf"),
                        device=logits.device,
                        dtype=logits.dtype,
                    )
                    expanded_logits[:, :, list(self.output_prune_map.values())] = logits
                else:
                    # (1, pruned_size) -> (1, original_size)
                    expanded_logits = torch.full(
                        [logits.shape[0], self.vocab_size],
                        float("-inf"),
                        device=logits.device,
                        dtype=logits.dtype,
                    )
                    expanded_logits[:, list(self.output_prune_map.values())] = logits
                logits = expanded_logits
        else:
            logits = h

        if attn_options_update is not None:
            return logits, attn_options_update

        return logits


def construct_transformer(model_args: ModelArgs) -> Transformer:
    """
    Construct a Transformer model from the given model arguments.
    """
    rope = Rope(model_args)
    if model_args.attention_type not in ATTENTION_REGISTRY:
        raise ValueError(
            f"Unknown attention type: {model_args.attention_type}. "
            f"Available: {list(ATTENTION_REGISTRY.keys())}"
        )
    layers = torch.nn.ModuleList()
    cls = ATTENTION_REGISTRY[model_args.attention_type]
    for layer_id in range(model_args.n_layers):
        # hybrid models define layer_types
        if model_args.layer_types and model_args.layer_types[layer_id] == "conv":
            from executorch.examples.models.lfm2.short_conv import ShortConvBlock

            assert (
                model_args.hidden_dim is not None
            ), "`hidden_dim` must be set in ModelArgs to construct a TransformerBlock."
            layers.append(
                ShortConvBlock(
                    dim=model_args.dim,
                    hidden_dim=model_args.hidden_dim,
                    norm_eps=model_args.norm_eps,
                )
            )
        else:
            attention = cls(
                model_args, layer_id, rope, **model_args.attention_kwargs
            )  # pyre-ignore[45]
            transformer_block = TransformerBlock(model_args, attention)
            layers.append(transformer_block)

    return Transformer(model_args, layers, rope)
