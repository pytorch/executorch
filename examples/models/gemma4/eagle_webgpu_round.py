# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import math
from typing import Any, cast

import torch

from executorch.examples.models.gemma4.mtp_qat_contract import (
    OFFICIAL_QAT_CENTROID_TOP_K,
    OFFICIAL_QAT_NUM_CENTROIDS,
    OFFICIAL_QAT_SELECTED_TOKEN_COUNT,
    OFFICIAL_QAT_TOKENS_PER_CENTROID,
    validate_qat_token_ordering,
)


def _selection_contract(
    centroid_top_k: int, selected_token_count: int
) -> dict[str, int]:
    if type(centroid_top_k) is not int or type(selected_token_count) is not int:
        raise ValueError("assistant selection dimensions must be integers")
    if centroid_top_k != OFFICIAL_QAT_CENTROID_TOP_K:
        raise ValueError("official QAT assistant requires centroid top-k=32")
    if selected_token_count != OFFICIAL_QAT_SELECTED_TOKEN_COUNT:
        raise ValueError("official QAT assistant requires 4096 selected tokens")
    return {
        "centroidTopK": centroid_top_k,
        "numCentroids": OFFICIAL_QAT_NUM_CENTROIDS,
        "selectedTokenCount": selected_token_count,
        "tokensPerCentroid": OFFICIAL_QAT_TOKENS_PER_CENTROID,
    }


def select_qat_centroids(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() != OFFICIAL_QAT_NUM_CENTROIDS:
        raise ValueError("QAT centroid scores must contain 2048 entries")
    return torch.topk(
        scores.reshape(-1), OFFICIAL_QAT_CENTROID_TOP_K, sorted=True
    ).indices


def validate_selected_destinations(
    token_ordering: torch.Tensor, selected_centroids: torch.Tensor
) -> torch.Tensor:
    evidence = validate_qat_token_ordering(token_ordering)
    if evidence["permutationExact"] is not True:
        raise ValueError("QAT token ordering must be an exact permutation")
    if selected_centroids.numel() != OFFICIAL_QAT_CENTROID_TOP_K:
        raise ValueError("QAT selection must contain 32 centroids")
    if selected_centroids.dtype not in (torch.int32, torch.int64):
        raise ValueError("QAT centroid indices must use an integer dtype")
    if torch.any(selected_centroids < 0) or torch.any(
        selected_centroids >= OFFICIAL_QAT_NUM_CENTROIDS
    ):
        raise ValueError("QAT centroid index out of range")
    if torch.unique(selected_centroids).numel() != OFFICIAL_QAT_CENTROID_TOP_K:
        raise ValueError("QAT selected centroids must be distinct")

    logical = token_ordering.reshape(
        OFFICIAL_QAT_NUM_CENTROIDS, OFFICIAL_QAT_TOKENS_PER_CENTROID
    )
    destinations = logical[selected_centroids.to(torch.int64)].reshape(-1)
    if destinations.numel() != OFFICIAL_QAT_SELECTED_TOKEN_COUNT or (
        torch.unique(destinations).numel() != destinations.numel()
    ):
        raise ValueError("QAT selected destinations must be pairwise distinct")
    return destinations


class K2LongestPrefixSelector(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "zero", torch.zeros((1,), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "one", torch.ones((1,), dtype=torch.float32), persistent=False
        )

    def forward(
        self,
        round_tokens: torch.Tensor,
        greedy_predictions: torch.Tensor,
        target_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        round_tokens_fp32 = round_tokens.to(torch.float32)
        greedy_fp32 = greedy_predictions.to(torch.float32)
        target_features_fp32 = target_features.to(torch.float32)
        candidates_fp32 = round_tokens_fp32.narrow(1, 1, 2)

        first_matches = (candidates_fp32[:, 0] - greedy_fp32[:, 0]) == 0.0
        second_matches = (candidates_fp32[:, 1] - greedy_fp32[:, 1]) == 0.0
        two = self.one + self.one
        accepted_count_fp32 = torch.where(
            first_matches,
            torch.where(second_matches, two, self.one),
            self.zero,
        )
        bonus_fp32 = torch.where(
            first_matches,
            torch.where(second_matches, greedy_fp32[:, 2], greedy_fp32[:, 1]),
            greedy_fp32[:, 0],
        )
        selected_feature = torch.where(
            first_matches,
            torch.where(
                second_matches,
                target_features_fp32[:, 2],
                target_features_fp32[:, 1],
            ),
            target_features_fp32[:, 0],
        )
        return (
            accepted_count_fp32.to(torch.long),
            bonus_fp32.to(torch.long),
            candidates_fp32.to(torch.long),
            selected_feature,
        )


class Gemma4K2Target(torch.nn.Module):
    def __init__(self, text_model: Any) -> None:
        super().__init__()
        self.text_model: Any = text_model
        donors: dict[bool, list[int]] = {False: [], True: []}
        for index, layer in enumerate(text_model.self_decoder.layers):
            attention = layer.self_attn
            if getattr(attention, "is_kv_donor_layer", False):
                if attention.kv_cache is None:
                    raise ValueError("Gemma4 K=2 donor has no KV cache")
                donors[bool(attention.is_sliding)].append(index)
        if any(len(indices) != 1 for indices in donors.values()):
            raise ValueError(f"Gemma4 K=2 donor topology mismatch: {donors}")
        if donors != {False: [14], True: [13]}:
            raise ValueError(f"Gemma4 K=2 donor layer mismatch: {donors}")
        self.full_donor_index = donors[False][0]
        self.sliding_donor_index = donors[True][0]

    def donor_views(
        self, donor_length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        length = cast(int, donor_length[0, 0].item())
        torch._check_is_size(length)
        torch._check(length >= 2)
        full = self.text_model.self_decoder.layers[
            self.full_donor_index
        ].self_attn.kv_cache
        sliding = self.text_model.self_decoder.layers[
            self.sliding_donor_index
        ].self_attn.kv_cache
        return (
            full.k_cache.narrow(1, 0, length).transpose(1, 2),
            full.v_cache.narrow(1, 0, length).transpose(1, 2),
            sliding.k_cache.narrow(1, 0, length).transpose(1, 2),
            sliding.v_cache.narrow(1, 0, length).transpose(1, 2),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        input_pos: torch.Tensor,
        is_round: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode = cast(int, is_round[0].item())
        torch._check(mode >= 0, lambda: "mode must be 0 or 1")
        torch._check_is_size(mode)
        torch._check(mode <= 1, lambda: "mode must be 0 or 1")
        m = 1 + 2 * mode
        length = input_pos.size(0)
        torch._check(length >= m, lambda: "round input has too few rows")
        torch._check(
            length <= 3 + (1 - mode) * (self.text_model.config.max_seq_len - 3),
            lambda: "round input must have exactly three rows",
        )
        terminal_pos = cast(int, input_pos[-1].item())
        query_start_pos = terminal_pos - m + 1
        torch._check(query_start_pos >= 0, lambda: "positions out of capacity")
        torch._check(
            terminal_pos < self.text_model.config.max_seq_len,
            lambda: "positions out of capacity",
        )

        hidden_states, per_layer_inputs, shared_kv = self.text_model.self_decoder(
            input_ids=input_ids,
            input_pos=input_pos,
            inputs_embeds=None,
        )
        hidden_length = hidden_states.size(1)
        per_layer_length = per_layer_inputs.size(2)
        torch._check(
            hidden_length == length,
            lambda: "self decoder hidden length must match input length",
        )
        torch._check(
            per_layer_length == length,
            lambda: "self decoder per-layer length must match input length",
        )
        hidden_states = torch.narrow(hidden_states, 1, hidden_length - m, m)
        per_layer_inputs = torch.narrow(per_layer_inputs, 2, per_layer_length - m, m)
        hidden_states = self.text_model.cross_decoder(
            hidden_states=hidden_states,
            per_layer_inputs=per_layer_inputs,
            shared_kv=shared_kv,
            input_pos=input_pos,
            query_start_pos=query_start_pos,
        )
        all_features = self.text_model.norm(hidden_states)
        all_logits = self.text_model.lm_head(all_features)
        features = torch.ops.aten.slice.Tensor(
            torch.cat((all_features, all_features, all_features), dim=1),
            1,
            0,
            3,
        )
        all_greedy = torch.argmax(all_logits, dim=-1).to(torch.float32)
        greedy = torch.ops.aten.slice.Tensor(
            torch.cat((all_greedy, all_greedy, all_greedy), dim=1),
            1,
            0,
            3,
        ).to(torch.long)
        return greedy, features


class K2GPUResidentRound(torch.nn.Module):
    def __init__(
        self,
        target: Any,
        embed_tokens: Any,
        assistant: Any,
        hidden_size: int,
        max_input_len: int,
        max_donor_len: int,
        embed_scale: float,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or max_input_len < 3 or max_donor_len < 2:
            raise ValueError("invalid K=2 combined-round dimensions")
        if not math.isfinite(embed_scale) or embed_scale <= 0.0:
            raise ValueError("invalid Gemma 4 target embedding scale")
        self.target: Any = target
        self.embed_tokens: Any = embed_tokens
        self.assistant: Any = assistant
        self.selector = K2LongestPrefixSelector()
        self.max_donor_len = max_donor_len
        self.embed_scale = embed_scale
        self.register_buffer(
            "seed_feature",
            torch.zeros((1, 1, 1, hidden_size), dtype=torch.float32),
        )
        self.register_buffer(
            "round_tail",
            torch.zeros((1, max_input_len - 3), dtype=torch.float32),
            persistent=False,
        )

    def update_seed_feature(self, next_feature: torch.Tensor) -> torch.Tensor:
        torch.ops.llama.update_cache.default(
            next_feature.unsqueeze(2), self.seed_feature, 0
        )
        return self.seed_feature

    def forward(
        self,
        input_ids: torch.Tensor,
        input_pos: torch.Tensor,
        is_round: torch.Tensor,
        donor_length: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        donor_k = cast(int, donor_length[0, 0].item())
        torch._check(donor_k >= 2, lambda: "donor length out of capacity")
        torch._check_is_size(donor_k)
        torch._check(
            donor_k <= self.max_donor_len,
            lambda: "donor length out of capacity",
        )
        round_condition = is_round.to(torch.float32) == 1.0
        full_k, full_v, sliding_k, sliding_v = self.target.donor_views(donor_length)
        first_position_ids = donor_length - 1
        seed_token = input_ids.narrow(1, 0, 1)
        seed_embedding = self.embed_tokens(seed_token) * self.embed_scale
        first_input = torch.cat((seed_embedding, self.seed_feature.squeeze(2)), dim=-1)
        first_logits, first_hidden = self.assistant(
            first_input,
            first_position_ids,
            full_k,
            full_v,
            sliding_k,
            sliding_v,
        )
        first_candidate = torch.argmax(first_logits, dim=-1)

        second_embedding = self.embed_tokens(first_candidate) * self.embed_scale
        second_input = torch.cat((second_embedding, first_hidden), dim=-1)
        second_logits, _ = self.assistant(
            second_input,
            donor_length,
            full_k,
            full_v,
            sliding_k,
            sliding_v,
        )
        second_candidate = torch.argmax(second_logits, dim=-1)
        round_padded = torch.cat(
            (
                seed_token.to(torch.float32),
                first_candidate.to(torch.float32),
                second_candidate.to(torch.float32),
                self.round_tail,
            ),
            dim=1,
        )
        round_tokens = round_padded.narrow(1, 0, input_ids.size(1))
        target_ids = torch.where(
            round_condition, round_tokens, input_ids.to(torch.float32)
        ).to(torch.long)
        target_greedy, target_features = self.target(target_ids, input_pos, is_round)
        matches, round_bonus, candidates, selected_feature = self.selector(
            round_padded, target_greedy, target_features
        )

        prefill_feature = target_features[:, -1:, :]
        round_feature = selected_feature.unsqueeze(1)
        next_feature = torch.where(round_condition, round_feature, prefill_feature)
        updated_seed_feature = self.update_seed_feature(next_feature)

        matches_fp32 = matches.to(torch.float32)
        output_matches = torch.where(
            round_condition, matches_fp32, self.selector.zero
        ).to(torch.long)
        output_bonus = torch.where(
            round_condition,
            round_bonus.view(1, 1).to(torch.float32),
            target_greedy[:, -1:].to(torch.float32),
        ).to(torch.long)
        state_probe = updated_seed_feature.narrow(3, 0, 1).reshape(1, 1).clone()
        return (
            candidates,
            target_greedy.to(torch.long),
            output_matches,
            output_bonus,
            state_probe,
        )


def _require_tensor_contract(
    value: Any,
    label: str,
    expected_shape: tuple[int | None, ...],
    expected_dtype: torch.dtype,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"K=2 {label} is not a tensor")
    if len(value.shape) != len(expected_shape):
        raise ValueError(f"K=2 {label} rank mismatch: {tuple(value.shape)}")
    for actual, expected in zip(value.shape, expected_shape):
        if expected is not None and actual != expected:
            raise ValueError(f"K=2 {label} shape mismatch: {tuple(value.shape)}")
    if value.dtype != expected_dtype:
        raise ValueError(f"K=2 {label} dtype mismatch: {value.dtype}")


def _tensor_meta(value: object) -> torch.Tensor | None:
    if not isinstance(value, torch.fx.Node):
        return None
    tensor = value.meta.get("val")
    return tensor if isinstance(tensor, torch.Tensor) else None


def _normalize_mutation_target(target: object) -> str:
    value = str(target)
    if value.endswith("seed_feature"):
        return "seed_feature"
    marker = "self_decoder.layers."
    if marker not in value:
        return value
    return value[value.index(marker) :]


def _expected_mutation_contract(max_donor_len: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = [
        {
            "logicalTarget": "seed_feature",
            "role": "nextFeatureSeed",
            "shape": [1, 1, 1, 1536],
            "logicalLayout": "BSHD",
            "logicalDimOrder": [0, 1, 2, 3],
            "vulkanSourceStorage": "BUFFER",
            "vulkanDestinationStorage": "TEXTURE_3D",
        }
    ]
    for layer in range(15):
        head_dim = 512 if layer in {4, 9, 14} else 256
        for cache_kind in ("k_cache", "v_cache"):
            records.append(
                {
                    "logicalTarget": (
                        f"self_decoder.layers.{layer}.self_attn.kv_cache."
                        f"{cache_kind}"
                    ),
                    "role": "targetKvCache",
                    "layer": layer,
                    "cacheKind": cache_kind,
                    "shape": [1, max_donor_len, 1, head_dim],
                    "logicalLayout": "BSHD",
                    "logicalDimOrder": [0, 1, 2, 3],
                    "vulkanSourceStorage": "BUFFER",
                    "vulkanDestinationStorage": "BUFFER",
                }
            )
    return records


def _validate_mutation_contract(
    program: torch.export.ExportedProgram, max_donor_len: int
) -> list[dict[str, Any]]:
    signature = program.graph_signature
    output_specs = list(signature.output_specs)
    expected_kinds = ["BUFFER_MUTATION"] * 31 + ["USER_OUTPUT"] * 5
    actual_kinds = [spec.kind.name for spec in output_specs]
    if actual_kinds != expected_kinds:
        raise ValueError(f"K=2 output-spec order mismatch: {actual_kinds}")
    expected = _expected_mutation_contract(max_donor_len)
    mutations = output_specs[: len(expected)]
    actual_targets = [_normalize_mutation_target(spec.target) for spec in mutations]
    expected_targets = [str(record["logicalTarget"]) for record in expected]
    if actual_targets != expected_targets:
        raise ValueError(
            "K=2 mutation target order mismatch: "
            f"{actual_targets} != {expected_targets}"
        )
    graph_nodes = {node.name: node for node in program.graph.nodes}
    for spec, record in zip(mutations, expected):
        name = getattr(spec.arg, "name", None)
        node = graph_nodes.get(name)
        if node is None:
            raise ValueError(f"K=2 mutation output is missing: {name}")
        _require_tensor_contract(
            node.meta.get("val"),
            str(record["logicalTarget"]),
            tuple(record["shape"]),
            torch.float32,
        )
    return expected


def _donor_view_contract() -> list[dict[str, object]]:
    return [
        {"role": "fullK", "layer": 14, "cacheKind": "k_cache", "layout": "BHKD"},
        {"role": "fullV", "layer": 14, "cacheKind": "v_cache", "layout": "BHKD"},
        {
            "role": "slidingK",
            "layer": 13,
            "cacheKind": "k_cache",
            "layout": "BHKD",
        },
        {
            "role": "slidingV",
            "layer": 13,
            "cacheKind": "v_cache",
            "layout": "BHKD",
        },
    ]


def _validate_seed_alias(
    program: torch.export.ExportedProgram,
) -> dict[str, object]:
    signature = program.graph_signature
    seed_inputs = [
        node
        for node in program.graph.nodes
        if node.op == "placeholder"
        and str(signature.inputs_to_buffers.get(node.name, "")).endswith("seed_feature")
    ]
    if len(seed_inputs) != 1:
        raise ValueError("K=2 seed-feature buffer alias mismatch")
    seed = seed_inputs[0]
    updates = [
        node
        for node in program.graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.llama.update_cache.default
        and len(node.args) == 3
        and node.args[1] is seed
    ]
    if len(updates) != 1 or updates[0].args[2] != 0:
        raise ValueError("K=2 seed-feature mutation binding mismatch")
    source = updates[0].args[0]
    _require_tensor_contract(
        _tensor_meta(source),
        "seed mutation source",
        (1, 1, 1, 1536),
        torch.float32,
    )
    if (
        not isinstance(source, torch.fx.Node)
        or source.target != torch.ops.aten.unsqueeze.default
        or not source.args
    ):
        raise ValueError("K=2 seed-feature source must be nextFeature unsqueeze")
    _require_tensor_contract(
        _tensor_meta(source.args[0]),
        "nextFeature",
        (1, 1, 1536),
        torch.float32,
    )
    return {
        "logicalSource": "nextFeature[1,1,1536]",
        "physicalDestination": "seed_feature[1,1,1,1536]",
        "mutation": "llama.update_cache.default",
    }


def validate_k2_round_abi(  # noqa: C901
    program: torch.export.ExportedProgram,
    *,
    max_input_len: int,
    max_donor_len: int,
) -> dict[str, Any]:
    signature = program.graph_signature
    expected_inputs = ("input_ids", "input_pos", "is_round", "donor_length")
    if tuple(signature.user_inputs) != expected_inputs:
        raise ValueError(
            f"K=2 user-input order mismatch: {tuple(signature.user_inputs)}"
        )
    graph_nodes = {node.name: node for node in program.graph.nodes}
    input_contracts = (
        ("input_ids", (1, None), torch.int64),
        ("input_pos", (None,), torch.int64),
        ("is_round", (1,), torch.int64),
        ("donor_length", (1, 1), torch.int64),
    )
    for name, shape, dtype in input_contracts:
        node = graph_nodes.get(name)
        if node is None or node.op != "placeholder":
            raise ValueError(f"K=2 missing user input: {name}")
        _require_tensor_contract(node.meta.get("val"), name, shape, dtype)
    input_ids = graph_nodes["input_ids"].meta["val"]
    input_pos = graph_nodes["input_pos"].meta["val"]
    if str(input_ids.shape[1]) != str(input_pos.shape[0]):
        raise ValueError("K=2 input_ids/input_pos dynamic dimensions differ")

    user_outputs = tuple(signature.user_outputs)
    if len(user_outputs) != 5:
        raise ValueError(f"K=2 user-output count mismatch: {len(user_outputs)}")
    output_contracts = (
        ((1, 2), torch.int64, "candidates"),
        ((1, 3), torch.int64, "target_greedy"),
        ((1,), torch.int64, "output_matches"),
        ((1, 1), torch.int64, "output_bonus"),
        ((1, 1), torch.float32, "state_probe"),
    )
    for name, (shape, dtype, label) in zip(user_outputs, output_contracts):
        node = graph_nodes.get(name)
        if node is None:
            raise ValueError(f"K=2 missing user output: {name}")
        _require_tensor_contract(node.meta.get("val"), label, shape, dtype)

    mutation_contract = _validate_mutation_contract(program, max_donor_len)

    operator_counts: dict[str, int] = {}
    for node in program.graph.nodes:
        if node.op == "call_function":
            target = str(node.target)
            operator_counts[target] = operator_counts.get(target, 0) + 1
    expected_operator_counts = {
        "aten.argmax.default": 3,
        "aten.scatter.src": 2,
        "aten.topk.default": 2,
        "llama.custom_sdpa.default": 43,
        "llama.update_cache.default": 31,
    }
    for target, expected in expected_operator_counts.items():
        if operator_counts.get(target, 0) != expected:
            raise ValueError(
                f"K=2 {target} count mismatch: "
                f"{operator_counts.get(target, 0)} != {expected}"
            )

    range_bounds: set[tuple[int, int]] = set()
    for value in program.range_constraints.values():
        try:
            bounds = (int(value.lower), int(value.upper))
        except (TypeError, ValueError, OverflowError):
            continue
        range_bounds.add(bounds)
    for expected_range in ((1, max_input_len), (2, max_donor_len)):
        if expected_range not in range_bounds:
            raise ValueError(f"K=2 missing dynamic range: {expected_range}")
    return {
        "bufferMutationCount": len(mutation_contract),
        "donorViewOrder": _donor_view_contract(),
        "inputOrder": [contract[0] for contract in input_contracts],
        "mutationOrder": mutation_contract,
        "operatorCounts": expected_operator_counts,
        "outputOrder": [contract[2] for contract in output_contracts],
        "seedMutationCount": 1,
        "stateAlias": _validate_seed_alias(program),
    }


def export_k2_round_program(
    module: K2GPUResidentRound, max_input_len: int = 64
) -> torch.export.ExportedProgram:
    if max_input_len < 3 or module.round_tail.numel() + 3 < max_input_len:
        raise ValueError("invalid K=2 round input bound")
    seq_len = torch.export.Dim("seq_len", min=1, max=max_input_len)
    return torch.export.export(
        module,
        (
            torch.ones((1, 3), dtype=torch.long),
            torch.arange(3, dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([[2]], dtype=torch.long),
        ),
        dynamic_shapes={
            "input_ids": {1: seq_len},
            "input_pos": {0: seq_len},
            "is_round": None,
            "donor_length": None,
        },
        strict=False,
    )


def _rewrite_negative_select_as_symint(
    program: torch.export.ExportedProgram,
) -> int:
    import executorch.backends.vulkan.custom_ops_lib  # noqa: F401

    graph = program.graph_module.graph
    item_targets = {
        torch.ops.aten.item.default,
        torch.ops.aten._local_scalar_dense.default,
    }
    replacements = 0
    for item_node in list(graph.nodes):
        if item_node.op != "call_function" or item_node.target not in item_targets:
            continue
        if len(item_node.args) != 1 or not isinstance(item_node.args[0], torch.fx.Node):
            continue
        select_node = item_node.args[0]
        if (
            select_node.op != "call_function"
            or select_node.target != torch.ops.aten.select.int
            or len(select_node.args) < 3
            or not isinstance(select_node.args[2], int)
            or select_node.args[2] >= 0
        ):
            continue
        source = select_node.args[0]
        if not isinstance(source, torch.fx.Node) or source.op != "placeholder":
            raise ValueError("K=2 negative select_as_symint source must be an input")
        source_dtype = getattr(source.meta.get("val"), "dtype", None)
        if source_dtype not in {torch.int32, torch.int64}:
            raise ValueError("K=2 negative select_as_symint requires integer input")
        with graph.inserting_before(item_node):
            replacement = graph.call_function(
                torch.ops.et_vk.select_as_symint.default,
                args=(source, select_node.args[1], select_node.args[2]),
            )
        replacement.meta = item_node.meta.copy()
        item_node.replace_all_uses_with(replacement)
        graph.erase_node(item_node)
        if not select_node.users:
            graph.erase_node(select_node)
        replacements += 1
    if replacements:
        program.graph_module.recompile()
    return replacements


def compose_k2_round_program(
    text_model: Any,
    embed_tokens: Any,
    assistant: Any,
    hidden_size: int,
    max_seq_len: int,
    embed_scale: float,
    max_input_len: int = 64,
    max_donor_len: int | None = None,
) -> torch.export.ExportedProgram:
    donor_bound = max_seq_len if max_donor_len is None else max_donor_len
    module = K2GPUResidentRound(
        target=Gemma4K2Target(text_model),
        embed_tokens=embed_tokens,
        assistant=assistant,
        hidden_size=hidden_size,
        max_input_len=max_input_len,
        max_donor_len=donor_bound,
        embed_scale=embed_scale,
    ).eval()
    program = export_k2_round_program(module, max_input_len)
    replacements = _rewrite_negative_select_as_symint(program)
    if replacements != 2:
        raise ValueError(
            "K=2 composition requires exactly two select_as_symint rewrites, "
            f"found {replacements}"
        )
    program = program.run_decompositions({})
    abi = validate_k2_round_abi(
        program,
        max_input_len=max_input_len,
        max_donor_len=donor_bound,
    )
    program.graph_module.meta["gemma4K2Abi"] = abi
    return program
