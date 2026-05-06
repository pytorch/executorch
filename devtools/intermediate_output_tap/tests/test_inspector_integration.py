# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Integration test: run the full pipeline (export -> tap -> lower with XNNPACK
-> strip -> generate_etrecord -> to_executorch -> runtime) and feed the flat
runtime outputs + (post-strip) TapSpecs to
Inspector.calculate_numeric_gap_from_taps. Verify the returned DataFrame has
rows aligned by debug_handle.

KEY DESIGN POINTS:
1. ETRecord generation MUST happen AFTER `strip_taps_` so the snapshot of the
   edge program contains no `tap.Tensor` nodes (which the EXIR serializer
   can't handle).
2. `strip_taps_(edge, tap_specs=specs)` returns updated specs whose
   `reducer_node_name` is set to the post-strip reducer terminal node name.
   Inspector uses that name to look up the post-roundtrip `debug_handle` —
   FX node names survive ETRecord serialization, debug_handle values do not.
"""

import os
import sys
import tempfile
import unittest

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.devtools import generate_etrecord, Inspector
from executorch.devtools.intermediate_output_tap import (
    DEFAULT_STATS,
    format_tap_dataframe,
    select_by_op_type,
    strip_taps_,
    tap_intermediate_outputs,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime, Verification
from torch.export import export


class _MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(8, 16)
        self.l2 = torch.nn.Linear(16, 4)

    def forward(self, x):
        return self.l2(self.l1(x).relu())


@unittest.skipIf(sys.platform.startswith("win"), "ExecuTorch runtime not available on Windows")
class InspectorIntegrationTest(unittest.TestCase):
    def test_calculate_numeric_gap_from_taps(self):
        model = _MLP()
        example_inputs = (torch.randn(2, 8),)

        ep = export(model, example_inputs, strict=True)
        ep_t, specs = tap_intermediate_outputs(
            ep,
            selector=select_by_op_type("aten.linear.default"),
            reducer=DEFAULT_STATS,
        )
        # Do NOT pass generate_etrecord=True — we'd snapshot the EP while it
        # still has tap.Tensor nodes (unserializable).
        edge = to_edge_transform_and_lower(
            ep_t,
            partitioner=[XnnpackPartitioner()],
        )
        # strip_taps_ with tap_specs returns updated specs whose
        # reducer_node_name points at the post-strip reducer terminal node.
        specs = strip_taps_(edge, tap_specs=specs)
        et_program = edge.to_executorch()

        with tempfile.TemporaryDirectory() as temp_dir:
            pte_path = os.path.join(temp_dir, "model.pte")
            et_program.save(pte_path)

            # ETRecord generated AFTER strip — the edge program is now
            # serializable. Don't pass exported_program: Inspector falls back
            # to the edge dialect program for AOT capture.
            etrecord_path = os.path.join(temp_dir, "etrecord.bin")
            generate_etrecord(
                etrecord_path,
                edge_dialect_program=edge,
                executorch_program=et_program,
            )

            rt = Runtime.get()
            program = rt.load_program(
                pte_path,
                verification=Verification.Minimal,
                enable_etdump=True,
                debug_buffer_size=1024 * 1024,
            )
            method = program.load_method("forward")
            flat_outputs = method.execute(list(example_inputs))

            etdump_path = os.path.join(temp_dir, "etdump.etdp")
            debug_buffer_path = os.path.join(temp_dir, "debug_buffer.bin")
            program.write_etdump_result_to_file(etdump_path, debug_buffer_path)
            if not os.path.exists(etdump_path):
                self.skipTest(
                    "Event tracer not enabled. Run with "
                    "--config executorch.event_tracer_enabled=true"
                )

            inspector = Inspector(
                etdump_path=etdump_path,
                etrecord=etrecord_path,
                debug_buffer_path=debug_buffer_path,
            )
            inspector._etrecord._representative_inputs = list(example_inputs)
            df = inspector.calculate_numeric_gap_from_taps(
                flat_runtime_outputs=flat_outputs,
                tap_specs=specs,
                distance="MSE",
            )
            # Print friendly per-tap view to stdout (visible via --print-passing-details).
            friendly = format_tap_dataframe(df, specs)
            import pandas as _pd
            with _pd.option_context(
                "display.max_columns", None,
                "display.width", 240,
                "display.max_colwidth", 30,
                "display.float_format", "{:.4g}".format,
            ):
                print("\n=== Inspector.calculate_numeric_gap_from_taps (friendly) ===")
                print(friendly.to_string())

        self.assertGreater(len(df), 0, "expected at least one tap row in DataFrame")
        for col in ("aot_ops", "runtime_ops", "gap"):
            self.assertIn(col, df.columns)
        for _, row in df.iterrows():
            self.assertIsNotNone(row["aot_ops"])
            self.assertIsNotNone(row["runtime_ops"])
            gap = row["gap"]
            if isinstance(gap, list):
                gap = gap[0] if gap else 0.0
            self.assertGreaterEqual(float(gap), 0.0)
