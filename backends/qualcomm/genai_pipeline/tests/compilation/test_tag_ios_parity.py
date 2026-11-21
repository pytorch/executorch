# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parity between the pipeline's decoder tagger and the legacy ``_tag_ios``.

The two implementations must agree until the legacy flow is removed in Phase 3,
when this file is deleted with it.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from executorch.backends.qualcomm._passes import TagQuantIO
from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
)
from executorch.backends.qualcomm.genai_pipeline.compilation.pass_policy import (
    build_pass_config,
)
from executorch.backends.qualcomm.genai_pipeline.control_args import ControlArgs
from executorch.backends.qualcomm.genai_pipeline.model_lookup import get_model_layout
from executorch.backends.qualcomm.utils.constants import (
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
)
from executorch.exir.dialects._ops import ops as exir_ops

_DTYPES = {"kv_type": torch.uint8, "io_type": torch.uint16}

_META = {
    "get_ar_len": 1,
    "get_head_dim": 64,
    "get_max_batch_size": 1,
    "get_max_context_len": 128,
    "get_n_layers": 4,
    "get_vocab_size": 32000,
}

_GEMMA4_META = {**_META, "get_global_head_dim": 512, "get_n_self_layers": 3}


def _make_node(shape, op="call_function", name="node", target=None, users=()):
    """Create a node double whose shape and users both taggers can inspect."""
    node = MagicMock(name=name)
    node.op = op
    node.name = name
    node.target = target if target is not None else MagicMock(name=f"{name}_target")
    node.meta = {"val": torch.zeros(shape)}
    node.users = {user: None for user in users}
    return node


def _make_nodes():
    """One node per legacy tagging rule, plus nodes that match none."""
    output = _make_node((1,), op="output")
    fallback = _make_node((1, 1, 8), target=exir_ops.edge.llama.fallback.default)
    trace = _make_node((3, 7))
    trace.meta["stack_trace"] = "x = per_layer_inputs[i]"
    donor = _make_node((3, 7))
    donor.meta["from_node"] = [SimpleNamespace(name="full_k_0", from_node=None)]
    return {
        "kv_input": _make_node(
            (1, 1, 64, 128),
            op="placeholder",
            name="args_0",
            users=(_make_node((1, 1, 64, 128)),),
        ),
        "logits_input": _make_node((1, 1, 32000), op="placeholder"),
        "mask_input": _make_node((1, 1, 1, 128), op="placeholder"),
        "kv_output": _make_node((1, 1, 1, 64), users=(output,)),
        "global_kv_output": _make_node((1, 1, 1, 512), users=(output,)),
        "logits_output": _make_node((1, 1, 32000), users=(output,)),
        "shard_boundary": _make_node((1, 1, 8), users=(fallback,)),
        "rope_slice": _make_node((1, 32), target=exir_ops.edge.aten.select.int),
        "global_rope_slice": _make_node((1, 256), target=exir_ops.edge.aten.select.int),
        "per_layer_input": trace,
        "donor_kv": donor,
        "unrelated": _make_node((3, 7)),
    }


def _legacy_tag_ios(meta, decoder_model):
    """Bind the legacy ``TextDecoder._tag_ios`` to a minimal stub decoder."""
    from executorch.examples.qualcomm.oss_scripts.llama.wrappers.llm_wrappers import (
        TextDecoder,
    )

    head_dims = {meta["get_head_dim"]}
    if decoder_model == "gemma4-e2b":
        head_dims.add(meta["get_global_head_dim"])
    kv_cache_shape = set()
    for dim in head_dims:
        kv_cache_shape |= {
            (dim, meta["get_max_context_len"]),
            (meta["get_max_context_len"], dim),
            (dim, meta["get_ar_len"]),
            (meta["get_ar_len"], dim),
        }
    stub = SimpleNamespace(
        meta=meta,
        control_args=SimpleNamespace(decoder_model=decoder_model),
        io_shape={
            (meta["get_max_batch_size"], meta["get_ar_len"], meta["get_vocab_size"])
        },
        kv_cache_shape=kv_cache_shape,
    )
    return lambda node: TextDecoder._tag_ios(stub, node, fixed_point_type=_DTYPES)


def _pipeline_tag_ios(meta, layout):
    """Return the pipeline's decoder tagger for the same graph."""
    _, passes_job = build_pass_config(
        artifact_key=ARTIFACT_TEXT_DECODER,
        meta=meta,
        control_args=ControlArgs(),
        layout=layout,
        quant_io_dtypes=_DTYPES,
    )
    return passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
        "get_quant_io_dtype_fn"
    ]


class TestTagIosParity(unittest.TestCase):

    def test_matches_legacy_for_every_rule(self):
        """Both taggers return the same dtype for each node, per model."""
        cases = {
            "llama3_2-1b_instruct": _META,
            "gemma4-e2b": _GEMMA4_META,
        }
        for model, meta in cases.items():
            legacy = _legacy_tag_ios(meta, model)
            pipeline = _pipeline_tag_ios(meta, get_model_layout(model))
            for name, node in _make_nodes().items():
                with self.subTest(model=model, node=name):
                    self.assertEqual(pipeline(node), legacy(node))


if __name__ == "__main__":
    unittest.main()
