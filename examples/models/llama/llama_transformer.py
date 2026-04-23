# @lint-ignore-every LICENSELINT
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

# Please refer to README.md in the same folder for more information.

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from executorch.examples.models.llama.attention import (
    Attention,
    ATTENTION_REGISTRY,
    AttentionSkip,
    ForwardOptions,
)
from executorch.examples.models.llama.feed_forward import FeedForward, LoRAFeedForward
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.norm import RMSNorm
from executorch.examples.models.llama.rope import Rope
from torch import nn


def _is_kv_donor_layer(
    layer_idx: int, n_layers: int, num_kv_shared_layers: int
) -> bool:
    """Check if this layer donates K/V to later YOCO shared layers.

    A donor layer is the last non-shared layer before KV sharing starts.
    The donor is the layer immediately before the first KV-shared layer.
    """
    if num_kv_shared_layers <= 0:
        return False
    first_shared = n_layers - num_kv_shared_layers
    if first_shared <= 0:
        return False
    return layer_idx == first_shared - 1


def _is_kv_shared_layer(
    layer_idx: int, n_layers: int, num_kv_shared_layers: int
) -> bool:
    """Check if this layer uses shared K/V from a donor layer (YOCO)."""
    if num_kv_shared_layers <= 0:
        return False
    first_shared = n_layers - num_kv_shared_layers
    return layer_idx >= first_shared and first_shared > 0


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
    def __init__(
        self, args: ModelArgs, attention: Attention, mlp_type: str = "default"
    ):
        """
        Transformer block with support for pre-norm and post-norm.
        Args:
            args (ModelArgs): model configuration parameters.
            attention (Attention): attention object to use in the transformer
                block. See `attention.py` for types of attention. Make sure
                the attention type is registered in the ATTENTION_REGISTRY.
            mlp_type (str): MLP type for this layer. "default" for standard
                FFN, "skip" for no FFN block.
        """
        super().__init__()
        self.use_kv_cache = args.use_kv_cache
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.head_dim
        self.attention = attention
        self.mlp_type = mlp_type.lower()

        assert (
            args.hidden_dim is not None
        ), "`hidden_dim` must be set in ModelArgs to construct a TransformerBlock."
        if self.mlp_type == "skip":
            pass  # No FFN block for this layer
        elif args.moe:
            self.block_sparse_moe = MOEFeedForward(args)
        elif args.target_modules is not None and (
            "down_proj" in args.target_modules
            or "up_proj" in args.target_modules
            or "gate_proj" in args.target_modules
        ):
            self.feed_forward = LoRAFeedForward(args.dim, args.hidden_dim, args)
        else:
            self.feed_forward = FeedForward(dim=args.dim, hidden_dim=args.hidden_dim)

        if isinstance(self.attention, AttentionSkip):
            self.attention_norm = nn.Identity()
        else:
            self.attention_norm = RMSNorm(
                args.dim,
                eps=args.norm_eps,
                add_unit_offset=args.rms_norm_add_unit_offset,
            )
        if self.mlp_type != "skip":
            self.ffn_norm = RMSNorm(
                args.dim,
                eps=args.norm_eps,
                add_unit_offset=args.rms_norm_add_unit_offset,
            )

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
        mlp_type = "default"
        if args.mlp_type is not None and layer_id < len(args.mlp_type):
            mlp_type = args.mlp_type[layer_id]
        cls = ATTENTION_REGISTRY[args.attention_type]
        attention = cls(args, layer_id, rope, **args.attention_kwargs)
        return TransformerBlock(args, attention, mlp_type=mlp_type)

    def forward(self, x, freqs_cos, freqs_sin, attn_options: ForwardOptions):  # x: 1xN
        h, attn_options_update = self.attention(
            self.attention_norm(x), freqs_cos, freqs_sin, **attn_options
        )
        if not isinstance(self.attention, AttentionSkip):
            h = x + h

        if self.mlp_type == "skip":
            out = h
        elif hasattr(self, "block_sparse_moe"):
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
        self.norm = RMSNorm(
            params.dim,
            eps=params.norm_eps,
            add_unit_offset=params.rms_norm_add_unit_offset,
        )
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
        # YOCO (You Only Cache Once) KV sharing configuration.
        self.num_kv_shared_layers = params.num_kv_shared_layers

    def _forward_layers(
        self,
        h: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        attn_options_: Dict,
        seqlen: int,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """Run transformer layers with YOCO KV sharing support."""
        attn_options_update = None
        shared_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        is_prefill = seqlen > 1
        for layer_idx, layer in enumerate(self.layers):
            is_shared = _is_kv_shared_layer(
                layer_idx, self.n_layers, self.num_kv_shared_layers
            )

            if is_shared and is_prefill:
                continue

            if is_shared:
                donor_idx = self.n_layers - self.num_kv_shared_layers - 1
                if donor_idx in shared_kv:
                    attn_options_["shared_kv"] = shared_kv[donor_idx]

            h, attn_options_update = layer(h, freqs_cos, freqs_sin, attn_options_)

            if _is_kv_donor_layer(layer_idx, self.n_layers, self.num_kv_shared_layers):
                assert (
                    attn_options_update is not None
                    and "kv_to_share" in attn_options_update
                ), f"Donor layer {layer_idx} must produce kv_to_share"
                shared_kv[layer_idx] = attn_options_update["kv_to_share"]

            if attn_options_update is not None:
                attn_options_.update(**attn_options_update)

            attn_options_.pop("shared_kv", None)

        # Remove YOCO-internal kv_to_share before returning
        if attn_options_update is not None:
            attn_options_update.pop("kv_to_share", None)
            if not attn_options_update:
                attn_options_update = None

        return h, attn_options_update

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

        attn_options_ = attn_options.copy() if attn_options is not None else {}

        h, attn_options_update = self._forward_layers(
            h, freqs_cos, freqs_sin, attn_options_, seqlen
        )

        if not self.generate_full_logits:
            pos = attn_options.get("last_valid_token_pos", -1)
            h = h[:, pos, :]

        h = self.norm(h)

        if self.apply_output:
            logits = self.output(h)

            if self.output_prune_map is not None:
                if self.generate_full_logits:
                    expanded_logits = torch.full(
                        [logits.shape[0], logits.shape[1], self.vocab_size],
                        float("-inf"),
                        device=logits.device,
                        dtype=logits.dtype,
                    )
                    expanded_logits[:, :, list(self.output_prune_map.values())] = logits
                else:
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
        elif (
            model_args.layer_types
            and model_args.layer_types[layer_id] == "skip_attention"
        ):
            attention = AttentionSkip()
            transformer_block = TransformerBlock(model_args, attention)
            layers.append(transformer_block)
        elif (
            model_args.layer_types
            and model_args.layer_types[layer_id] == "linear_attention"
        ):
            linear_cls = ATTENTION_REGISTRY.get("gated_deltanet")
            if linear_cls is None:
                raise ValueError(
                    "Unknown attention type: gated_deltanet. "
                    f"Available: {list(ATTENTION_REGISTRY.keys())}"
                )
            attention = linear_cls(
                model_args, layer_id, rope, **model_args.attention_kwargs
            )
            transformer_block = TransformerBlock(model_args, attention)
            layers.append(transformer_block)
        else:
            attention = cls(
                model_args, layer_id, rope, **model_args.attention_kwargs
            )  # pyre-ignore[45]
            transformer_block = TransformerBlock(model_args, attention)
            layers.append(transformer_block)

    return Transformer(model_args, layers, rope)
