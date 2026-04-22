# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.fx import Node
from torch._subclasses.fake_tensor import FakeTensor
from executorch.exir.pass_base import ExportPass


class FixEmbeddingSymbolicShapes(ExportPass):
    """
    Re-propagates SymInt through aten.embedding.default output metadata
    after register_lstm_while_loop_decomposition concretizes it.
    """

    def call_operator(self, op, args, kwargs, meta):
        result = super().call_operator(op, args, kwargs, meta)

        if op is not torch.ops.aten.embedding.default:
            return result

        indices_node: Node = args[1]
        indices_fake = indices_node.meta.get("val")
        result_fake = result.node.meta.get("val")

        if indices_fake is None or result_fake is None:
            return result

        embed_dim = result_fake.shape[-1]
        indices_shape = indices_fake.shape
        expected_shape = (*indices_shape, embed_dim)

        needs_patch = any(
            isinstance(sym, torch.SymInt) and not isinstance(conc, torch.SymInt)
            for sym, conc in zip(expected_shape, result_fake.shape)
        )

        if not needs_patch:
            return result

        fake_mode = indices_fake.fake_mode
        with fake_mode:
            corrected = fake_mode.fake_tensor_converter.from_meta_and_device(
                torch.empty(expected_shape, dtype=result_fake.dtype),
                result_fake.device,
            )

        result.node.meta["val"] = corrected
        return result
