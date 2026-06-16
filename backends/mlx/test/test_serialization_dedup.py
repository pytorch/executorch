#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Serializer string-dedup regression test.

MetalKernelNode ``source``/``header`` blobs are large and repeated once per
layer. The serializer routes every string through ``_shared_string`` so
identical text is written into the FlatBuffer exactly once (multiple fields
share a single offset). The loader then interns those shared offsets into one
``std::shared_ptr<const std::string>`` per unique blob, so this dedup also
shrinks runtime memory for newly-produced ``.pte`` files.

This test pins the serializer half of that behavior.
"""

import unittest

from executorch.backends.mlx.serialization.mlx_graph_schema import (
    Instruction,
    InstructionChain,
    IntOrVid,
    MetalKernelNode,
    MLXGraph,
    Tid,
)
from executorch.backends.mlx.serialization.mlx_graph_serialize import (
    serialize_mlx_graph,
)


def _graph(nodes):
    chain = InstructionChain(instructions=[Instruction(op=n) for n in nodes])
    return MLXGraph(
        instruction_chains=[chain],
        version="test",
        input_map=[],
        output_map=[],
        mutable_buffer_map=[],
        named_slots=[],
        tensor_meta=[],
    )


def _kernel(source, header=None):
    return MetalKernelNode(
        name="gguf_q6k_matmul",
        source=source,
        inputs=[Tid(0)],
        outputs=[Tid(1)],
        grid=[IntOrVid(literal=1)],
        threadgroup=[IntOrVid(literal=1)],
        header=header,
        input_names=["x"],
        output_names=["out"],
    )


class TestSerializationStringDedup(unittest.TestCase):
    def test_identical_source_header_written_once(self):
        source = "KERNEL_SOURCE_MARKER_" + "x" * 2000
        header = "KERNEL_HEADER_MARKER_" + "y" * 2000

        nodes = [_kernel(source, header) for _ in range(5)]
        buf = serialize_mlx_graph(_graph(nodes))

        self.assertEqual(buf.count(source.encode()), 1)
        self.assertEqual(buf.count(header.encode()), 1)

    def test_distinct_sources_not_merged(self):
        base = "KERNEL_SOURCE_MARKER_" + "x" * 2000
        nodes = [_kernel(base + str(i)) for i in range(3)]
        buf = serialize_mlx_graph(_graph(nodes))

        # Each distinct source must still appear (the common prefix appears once
        # per distinct string since the suffixes differ).
        self.assertEqual(buf.count(base.encode()), 3)


if __name__ == "__main__":
    unittest.main()
