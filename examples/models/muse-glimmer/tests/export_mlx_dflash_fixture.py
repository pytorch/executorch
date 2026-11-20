# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small real MLX artifacts with controlled proposal rejection positions."""

import argparse
import copy
import json
import operator
from pathlib import Path
from unittest.mock import patch

import torch
from executorch import exir
from executorch.backends.mlx.pte_inspector import parse_mlx_payload
from executorch.examples.models.muse_glimmer.export import export_dflash
from executorch.examples.models.muse_glimmer.loaders.dflash_loader import (
    load_dflash_gguf,
)
from executorch.examples.models.muse_glimmer.model.dflash_model import (
    DFlashConfig,
    DFlashDraftModel,
    MuseGlimmerWithDFlash,
)
from executorch.examples.models.muse_glimmer.model.model import (
    MuseGlimmerConfig,
    MuseGlimmerModel,
    RMSNorm,
)
from executorch.examples.models.muse_glimmer.tests.generate_dflash_gguf_fixture import (
    write_dflash_gguf,
)
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.lowered_backend_module import LoweredBackendModule


def write_draft_reference(directory, target, draft, config, draft_config):
    eager_draft = (
        DFlashDraftModel(draft_config, max_context_length=config.max_seq_len)
        .eval()
        .requires_grad_(False)
    )
    eager_draft.load_state_dict(draft.state_dict(), assign=True)
    reference = (
        MuseGlimmerWithDFlash(copy.deepcopy(target), eager_draft, config, draft_config)
        .eval()
        .to(torch.float16)
    )
    references = [reference, copy.deepcopy(reference)]
    positions = [0, 0]
    block = draft_config.block_size
    hidden_size = draft_config.n_target_layers * draft_config.dim
    steps = []
    with torch.no_grad():
        for owners in (
            ((0, 2), (1, 1)),
            ((1, 2), (0, 2)),
            ((0, 1), (2, 0)),
            ((2, 2), (1, 2), (0, 0)),
            ((1, 0), (0, 0), (2, 0)),
            ((3, 0), (1, 0), (0, 0), (2, 0)),
        ):
            clone = len(steps) in (2, 5)
            if clone:
                references.append(copy.deepcopy(references[0]))
                positions.append(positions[0])
            rows = []
            for owner, context in owners:
                context *= block // 4
                position = positions[owner]
                tokens = torch.full((1, block), draft_config.mask_token_id)
                tokens[0, 0] = position + 1 + owner
                hidden = torch.randn(1, context, hidden_size, dtype=torch.float16)
                ids, scores = references[owner].draft_forward(
                    tokens, hidden, torch.tensor([position])
                )
                assert torch.isfinite(scores).all() and scores.std() > 0
                rows.append(
                    {
                        "owner": owner,
                        "position": position,
                        "context": context,
                        "tokens": tokens.flatten().tolist(),
                        "hidden": hidden.flatten().tolist(),
                        "ids": ids.flatten().tolist(),
                        "scores": scores.flatten().tolist(),
                    }
                )
                positions[owner] += context
            steps.append({"clone": clone, "rows": rows})
    (directory / "draft_reference.json").write_text(
        json.dumps(
            {
                "block_size": block,
                "top_k": draft_config.selector_top_k,
                "hidden_size": hidden_size,
                "max_seq_len": config.max_seq_len,
                "layers": [
                    {
                        "window": draft_config.sliding_window if sliding else 0,
                        "kv_heads": draft_config.n_kv_heads,
                        "head_dim": draft_config.head_dim,
                    }
                    for sliding in draft_config.sliding_window_pattern
                ],
                "steps": steps,
            }
        )
    )


def export_quantized_fixture(directory: Path):
    torch.manual_seed(73)
    checkpoint = directory / "draft.gguf"
    write_dflash_gguf(checkpoint)
    draft, draft_config = load_dflash_gguf(
        str(checkpoint), max_seq_len=64, activation_dtype=torch.float16
    )
    config = MuseGlimmerConfig(
        dim=256,
        n_layers=2,
        n_heads=4,
        n_kv_heads=1,
        head_dim=128,
        ffn_dim_multiplier=1.0,
        vocab_size=draft_config.vocab_size,
        global_attn_cfg="[16,0]",
        max_seq_len=64,
    )
    target = MuseGlimmerModel(config).eval()
    target.activation_dtype = torch.float16
    with torch.no_grad():
        # Separate candidate logits while keeping predictions prompt-dependent.
        target.embed_tokens.weight[::2, 0] = 16
        target.embed_tokens.weight[1::2, 0] = -16
        target.lm_head.weight.zero_()
        target.lm_head.weight[:32, 0] = torch.arange(1, 33) / 8
        target.lm_head.weight[32:64, 0] = -torch.arange(1, 33) / 8
    write_draft_reference(directory, target, draft, config, draft_config)

    lower = exir.to_edge_transform_and_lower

    def checked_lower(*args, **kwargs):
        edge = lower(*args, **kwargs)
        graph = edge.exported_program("draft_forward").graph_module
        compute = [
            node.target
            for node in graph.graph.nodes
            if node.op == "call_function"
            and node.target not in (executorch_call_delegate, operator.getitem)
        ]
        assert not compute, f"Draft has non-delegated compute: {compute}"
        operations = []
        for module in graph.modules():
            if isinstance(module, LoweredBackendModule):
                assert module.backend_id == "MLXBackend"
                payload = parse_mlx_payload(module.processed_bytes)
                operations.extend(
                    instruction
                    for chain in payload["graph"]["instruction_chains"]
                    for instruction in chain["instructions"]
                )
        quantized_weights = {
            op["w"]["tid"]
            for op in operations
            if op["op_name"] == "QuantizedMatmulNode" and op["bits"] == 4
        }
        # Includes four convolution projections and the selector projection.
        assert len(quantized_weights) == 18
        assert any(
            op["op_name"] == "MetalKernelNode" and op["name"] == "gguf_q6k_matmul"
            for op in operations
        )
        gathers = 0
        for i, op in enumerate(operations):
            if op["op_name"] == "DequantizeNode":
                assert op["bits"] == 4
                for field, source in zip(
                    ("w", "scales", "biases"), operations[i - 3 : i]
                ):
                    assert source["op_name"] == "TakeNode"
                    assert source["out"] == op[field]
                gathers += 1
        assert gathers == 2, "Selector tables must dequantize only gathered rows"
        (directory / "draft_operations.json").write_text(json.dumps(operations))
        return edge

    with patch.object(exir, "to_edge_transform_and_lower", checked_lower):
        export_dflash._export_dflash_mlx(
            target,
            config,
            draft,
            draft_config,
            None,
            None,
            str(directory),
            config.max_seq_len,
            torch.float16,
            4,
            use_offgraph_kv_cache=True,
        )


class RejectAt(torch.nn.Module):
    def __init__(self, selector, step):
        super().__init__()
        self.selector = selector
        self.step = step

    def forward(self, candidate_ids, unary_logits, hidden_states, anchor_token_ids):
        scores = self.selector(
            candidate_ids, unary_logits, hidden_states, anchor_token_ids
        )
        positions = torch.arange(candidate_ids.shape[1], device=candidate_ids.device)
        reject = (positions[None, :, None] == self.step) & (candidate_ids == 8)
        return scores + reject[:, :, None].float() * 100


def export_fixture(
    directory: Path, reject_at: int, legacy: bool = False, controlled: bool = True
):
    torch.manual_seed(73)
    config = MuseGlimmerConfig(
        dim=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        ffn_dim_multiplier=1.0,
        vocab_size=200096,
        global_attn_cfg="[4,0]",
        max_seq_len=32,
    )
    if controlled:
        config.output_multiplier = 1.0
        config.output_soft_cap_temp = 64.0
    draft_config = DFlashConfig(
        dim=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        ffn_dim=32,
        vocab_size=config.vocab_size,
        block_size=4,
        target_layers=[0],
        max_seq_len=config.max_seq_len,
        sliding_window=3,
        sliding_window_pattern=[True, False],
        mask_token_id=0,
        conv_kernel_size=0 if legacy else 2,
        conv_group_size=0 if legacy else 8,
        selector_rank=0 if legacy else 4,
        selector_top_k=0 if legacy else 3,
    )
    target = MuseGlimmerModel(config).eval()
    target.activation_dtype = torch.float16
    draft = DFlashDraftModel(draft_config, max_context_length=32).eval()
    with torch.no_grad():
        for model in (target, draft):
            if controlled:
                for parameter in model.parameters():
                    parameter.zero_()
                for module in model.modules():
                    if isinstance(module, RMSNorm):
                        module.weight.fill_(1)
        if controlled:
            target.embed_tokens.weight.fill_(1)
            target.lm_head.weight[7].fill_(1)
            target.lm_head.weight[8].fill_(63 / 64)
            target.lm_head.weight[9].fill_(31 / 32)
        else:
            # Keep the production vocabulary ABI with a small active vocabulary;
            # FP16 near-ties among 200K random logits obscure lifecycle failures.
            target.lm_head.weight[64:].zero_()
    if controlled and not legacy:
        draft.candidate_selector = RejectAt(draft.candidate_selector, reject_at)

    directory.mkdir(parents=True, exist_ok=True)
    if not controlled:
        write_draft_reference(directory, target, draft, config, draft_config)

    original_constants = export_dflash._dflash_constant_methods

    def constants(**kwargs):
        methods = original_constants(**kwargs)
        if legacy:
            methods.pop("get_dflash_selector_top_k", None)
        return methods

    with patch.object(export_dflash, "_dflash_constant_methods", constants):
        export_dflash._export_dflash_mlx(
            target,
            config,
            draft,
            draft_config,
            None,
            None,
            str(directory),
            config.max_seq_len,
            torch.float16,
            4,
            use_offgraph_kv_cache=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for reject_at in (-1, 0, 1, 2):
        export_fixture(args.output / f"reject_{reject_at}", reject_at)
    export_fixture(args.output / "legacy", -1, legacy=True)
    export_fixture(args.output / "random", -1, controlled=False)

    export_quantized_fixture(args.output / "quantized")
