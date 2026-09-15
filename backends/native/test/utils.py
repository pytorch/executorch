# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for the native backend tests."""

import torch

from executorch.backends.native import get_default_compile_config
from executorch.backends.native.partitioner import NativePartitioner
from executorch.backends.native.passes import get_default_passes
from executorch.backends.native.serialization.schema import OpKind
from executorch.exir import to_edge, to_edge_transform_and_lower


def _transformed(model, example_inputs, passes):
    edge = to_edge(
        torch.export.export(model, example_inputs),
        compile_config=get_default_compile_config(),
    )
    return edge.transform(passes).exported_program()


def _lower(model, example_inputs):
    ep = torch.export.export(model, example_inputs)
    return to_edge_transform_and_lower(
        ep,
        transform_passes=get_default_passes(),
        partitioner=[NativePartitioner()],
        compile_config=get_default_compile_config(),
    )


def _get_delegate_blob(edge):
    et = edge.to_executorch()
    delegates = et.executorch_program.backend_delegate_data
    assert len(delegates) == 1, f"Expected 1 delegate blob, got {len(delegates)}"
    return bytes(delegates[0].data)


def _call_function_targets(graph):
    return [n.target for n in graph.nodes if n.op_kind == OpKind.CALL_FUNCTION]
