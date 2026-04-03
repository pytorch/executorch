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


def _get_kv_donor_layer_idx(
    layer_idx: int,
    n_layers: int,
    num_kv_shared_layers: int,
    layer_types: Optional[list] = None,
) -> Optional[int]:
    if not _is_kv_shared_layer(layer_idx, n_layers, num_kv_shared_layers):
        return None

    first_shared = n_layers - num_kv_shared_layers
    if first_shared <= 0:
        return None
    if layer_types is None:
        return first_shared - 1

    target_type = layer_types[layer_idx]
    for donor_idx in range(first_shared - 1, -1, -1):
        if layer_types[donor_idx] == target_type:
            return donor_idx
    return None


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
    def __init__(self, args: ModelArgs, attention: Attention, layer_id: int):
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
        self.layer_id = layer_id
        self.hidden_size_per_layer_input = args.hidden_size_per_layer_input

        assert (
            args.hidden_dim is not None
        ), "`hidden_dim` must be set in ModelArgs to construct a TransformerBlock."
        ffn_hidden_dim = args.hidden_dim
        if _is_kv_shared_layer(layer_id, args.n_layers, args.num_kv_shared_layers):
            if args.use_double_wide_mlp:
                ffn_hidden_dim *= 2
        if args.moe:
            self.block_sparse_moe = MOEFeedForward(args)
        elif args.target_modules is not None and (
            "down_proj" in args.target_modules
            or "up_proj" in args.target_modules
            or "gate_proj" in args.target_modules
        ):
            self.feed_forward = LoRAFeedForward(args.dim, ffn_hidden_dim, args)
        else:
            self.feed_forward = FeedForward(
                dim=args.dim,
                hidden_dim=ffn_hidden_dim,
                act_fn=args.act_fn.get_function(),
            )

        if isinstance(self.attention, AttentionSkip):
            self.attention_norm = nn.Identity()
        else:
            self.attention_norm = RMSNorm(
                args.dim,
                eps=args.norm_eps,
                add_unit_offset=args.rms_norm_add_unit_offset,
            )
        self.ffn_norm = RMSNorm(
            args.dim,
            eps=args.norm_eps,
            add_unit_offset=args.rms_norm_add_unit_offset,
        )
        self.post_attention_norm = (
            RMSNorm(
                args.dim,
                eps=args.norm_eps,
                add_unit_offset=args.rms_norm_add_unit_offset,
            )
            if args.post_attention_norm and not isinstance(self.attention, AttentionSkip)
            else None
        )
        self.post_ffn_norm = (
            RMSNorm(
                args.dim,
                eps=args.norm_eps,
                add_unit_offset=args.rms_norm_add_unit_offset,
            )
            if args.post_ffn_norm
            else None
        )
        if self.hidden_size_per_layer_input > 0:
            self.per_layer_act = args.act_fn.get_function()
            self.per_layer_input_gate = nn.Linear(
                args.dim, self.hidden_size_per_layer_input, bias=False
            )
            self.per_layer_projection = nn.Linear(
                self.hidden_size_per_layer_input, args.dim, bias=False
            )
            self.post_per_layer_input_norm = RMSNorm(
                args.dim,
                eps=args.norm_eps,
                add_unit_offset=args.rms_norm_add_unit_offset,
            )
        if args.attention_type == "gemma4_mha":
            self.register_buffer("layer_scalar", torch.ones(1))

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
        return TransformerBlock(args, attention, layer_id)

    def forward(
        self,
        x,
        freqs_cos,
        freqs_sin,
        attn_options: ForwardOptions,
        per_layer_input: Optional[torch.Tensor] = None,
    ):  # x: 1xN
        h, attn_options_update = self.attention(
            self.attention_norm(x), freqs_cos, freqs_sin, **attn_options
        )
        if not isinstance(self.attention, AttentionSkip):
            if self.post_attention_norm is not None:
                h = self.post_attention_norm(h)
            h = x + h

        if hasattr(self, "block_sparse_moe"):
            ffn_out = self.block_sparse_moe(self.ffn_norm(h))
        else:
            ffn_out = self.feed_forward(self.ffn_norm(h))
        if self.post_ffn_norm is not None:
            ffn_out = self.post_ffn_norm(ffn_out)
        out = h + ffn_out

        if per_layer_input is not None and self.hidden_size_per_layer_input > 0:
            residual = out
            out = self.per_layer_input_gate(out)
            out = self.per_layer_act(out)
            out = out * per_layer_input
            out = self.per_layer_projection(out)
            out = self.post_per_layer_input_norm(out)
            out = residual + out
        if hasattr(self, "layer_scalar"):
            out = out * self.layer_scalar
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
        self.embedding_scale_factor = params.embedding_scale_factor
        self.final_logit_softcapping = params.final_logit_softcapping
        self.hidden_size_per_layer_input = params.hidden_size_per_layer_input

        self.tok_embeddings = (
            nn.Embedding(params.vocab_size, params.dim)
            if self.apply_embedding
            else None
        )
        if self.hidden_size_per_layer_input > 0:
            assert params.vocab_size_per_layer_input is not None
            self.embed_tokens_per_layer = nn.Embedding(
                params.vocab_size_per_layer_input,
                params.n_layers * self.hidden_size_per_layer_input,
            )
            self.per_layer_embedding_scale_factor = (
                params.hidden_size_per_layer_input**0.5
            )
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_model_projection = nn.Linear(
                params.dim,
                params.n_layers * self.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_model_projection_scale = params.dim**-0.5
            self.per_layer_projection_norm = RMSNorm(
                self.hidden_size_per_layer_input,
                eps=params.norm_eps,
                add_unit_offset=params.rms_norm_add_unit_offset,
            )
        else:
            self.embed_tokens_per_layer = None
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
        per_layer_inputs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """Run transformer layers with YOCO KV sharing support."""
        attn_options_update = None
        shared_kv: Dict[int, Any] = {}
        is_prefill = seqlen > 1
        for layer_idx, layer in enumerate(self.layers):
            is_shared = _is_kv_shared_layer(
                layer_idx, self.n_layers, self.num_kv_shared_layers
            )

            if (
                is_shared
                and is_prefill
                and self.params.attention_type != "gemma4_mha"
            ):
                continue

            if is_shared:
                donor_idx = _get_kv_donor_layer_idx(
                    layer_idx,
                    self.n_layers,
                    self.num_kv_shared_layers,
                    self.params.layer_types,
                )
                if donor_idx in shared_kv:
                    attn_options_["shared_kv"] = shared_kv[donor_idx]

            layer_per_input = (
                per_layer_inputs[:, :, layer_idx, :]
                if per_layer_inputs is not None
                else None
            )
            h, attn_options_update = layer(
                h,
                freqs_cos,
                freqs_sin,
                attn_options_,
                per_layer_input=layer_per_input,
            )

            if attn_options_update is not None and "kv_to_share" in attn_options_update:
                kv_to_share = attn_options_update["kv_to_share"]
                if isinstance(kv_to_share, dict):
                    shared_kv.update(kv_to_share)
                else:
                    shared_kv[layer_idx] = kv_to_share

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
            if self.embedding_scale_factor != 1.0:
                h = h * self.embedding_scale_factor

        if attn_options is None:
            attn_options = {}
        seqlen = h.shape[1]
        freqs_cos, freqs_sin = self.rope.get_freqs(
            attn_options.get("input_pos"), seqlen
        )
        per_layer_inputs = None
        if self.hidden_size_per_layer_input > 0:
            per_layer_projection = (
                self.per_layer_model_projection(h) * self.per_layer_model_projection_scale
            )
            per_layer_projection = per_layer_projection.reshape(
                *h.shape[:-1],
                self.n_layers,
                self.hidden_size_per_layer_input,
            )
            per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

            if tokens is not None and self.embed_tokens_per_layer is not None:
                per_layer_inputs = (
                    self.embed_tokens_per_layer(tokens)
                    * self.per_layer_embedding_scale_factor
                ).reshape(
                    *tokens.shape, self.n_layers, self.hidden_size_per_layer_input
                )
                per_layer_inputs = (
                    per_layer_projection + per_layer_inputs
                ) * self.per_layer_input_scale
            else:
                per_layer_inputs = per_layer_projection

        attn_options_ = attn_options.copy() if attn_options is not None else {}

        h, attn_options_update = self._forward_layers(
            h, freqs_cos, freqs_sin, attn_options_, seqlen, per_layer_inputs
        )

        if not self.generate_full_logits:
            pos = attn_options.get("last_valid_token_pos", -1)
            h = h[:, pos, :]

        h = self.norm(h)

        if self.apply_output:
            logits = self.output(h)
            if self.final_logit_softcapping is not None:
                logits = logits / self.final_logit_softcapping
                logits = torch.tanh(logits)
                logits = logits * self.final_logit_softcapping

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
            transformer_block = TransformerBlock(model_args, attention, layer_id)
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
            transformer_block = TransformerBlock(model_args, attention, layer_id)
            layers.append(transformer_block)
        else:
            attention = cls(
                model_args, layer_id, rope, **model_args.attention_kwargs
            )  # pyre-ignore[45]
            transformer_block = TransformerBlock(model_args, attention, layer_id)
            layers.append(transformer_block)

    return Transformer(model_args, layers, rope)
