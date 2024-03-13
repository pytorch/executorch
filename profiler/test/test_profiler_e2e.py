# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end profiler tests.

This must be built and run with `buck2 -c executorch.prof_enabled=true`.
"""

import unittest

import torch

from executorch.exir import to_edge

from executorch.extension.pybindings.portable_lib import (
    _create_profile_block,
    _dump_profile_results,
    _load_for_executorch_from_buffer,
    _reset_profile_results,
)
from executorch.extension.pytree import tree_flatten
from executorch.profiler.fb.parse_profiler_results import profile_table
from executorch.profiler.parse_profiler_results import (
    deserialize_profile_results,
    profile_aggregate_framework_tax,
    profile_framework_tax_table,
)
from torch.export import export


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("a", 3 * torch.ones(2, 2, dtype=torch.float))
        self.register_buffer("b", 2 * torch.ones(2, 2, dtype=torch.float))

    def forward(self, x):
        a = torch.mul(self.a, x)
        b = torch.add(a, self.b)
        return b


class TestCustomOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        model = Module()
        inputs = (torch.ones(2, 2, dtype=torch.float),)

        # The serialized program file. This must live longer than cls.module,
        # because the C++ pybindings will have a pointer to it. But none of the
        # tests should need to touch it.
        cls.__buffer: bytes = to_edge(export(model, inputs)).to_executorch().buffer

        cls.module = _load_for_executorch_from_buffer(cls.__buffer)

        # pyre-fixme[16]: Module `pytree` has no attribute `tree_flatten`.
        cls.inputs_flattened, _ = tree_flatten(inputs)
        cls.module.run_method("forward", tuple(cls.inputs_flattened))
        prof_dump = _dump_profile_results()
        assert (
            len(prof_dump) > 0
        ), "prof_dump is empty; may need to build with `-c executorch.prof_enabled=true`"
        cls.prof_results, cls.mem_results = deserialize_profile_results(prof_dump)
        cls.expect_ops = ["native_call_add.out", "native_call_mul.out"]

    def test_profiler_new_block(self) -> None:
        block_names = ["block_1", "block_2"]
        _reset_profile_results()
        _create_profile_block(block_names[0])
        self.module.run_method("forward", tuple(self.inputs_flattened))
        _create_profile_block(block_names[1])
        self.module.run_method("forward", tuple(self.inputs_flattened))
        prof_dump = _dump_profile_results()
        self.assertGreater(
            len(prof_dump),
            0,
            "prof_dump is empty; may need to build with `-c executorch.prof_enabled=true`",
        )
        prof_results, mem_results = deserialize_profile_results(prof_dump)
        for i, (block_name_, _) in enumerate(prof_results.items()):
            self.assertTrue(block_names[i] == block_name_)
        self.assertEqual(len(prof_results), 2)

    def test_profiler_expected_ops(self) -> None:
        found_count = 0
        for block_name, prof_data_list in self.prof_results.items():
            for prof_event in prof_data_list:
                if prof_event.name in self.expect_ops:
                    found_count += 1
            self.assertTrue(block_name == "default")
        self.assertEqual(found_count, len(self.expect_ops))

    def test_profile_framework_tax(self) -> None:
        prof_agg_data = profile_aggregate_framework_tax(self.prof_results)
        for name, framework_tax in prof_agg_data.items():
            self.assertTrue(len(framework_tax.exec_time) == 1)
            self.assertTrue(len(framework_tax.kernel_and_delegate_time) == 1)
            self.assertTrue(len(framework_tax.framework_tax) == 1)
            self.assertTrue(float(framework_tax.framework_tax[0]) < 100)
            self.assertTrue(name == "default")

    def test_gen_profile_table(self) -> None:
        prof_table = profile_table(self.prof_results)
        found_count = 0
        for table in prof_table:
            for entry in table:
                for op in self.expect_ops:
                    found_count += 1 if op in entry.get_string() else 0
        self.assertEqual(found_count, len(self.expect_ops))

    def test_gen_profile_framework_tax_table(self) -> None:
        prof_agg_data = profile_aggregate_framework_tax(self.prof_results)
        prof_framework_tax_table = profile_framework_tax_table(prof_agg_data)
        expected_entries = [
            "Model execution time",
            "Time spent in kernels",
            "Framework tax",
        ]
        found_count = 0
        for table in prof_framework_tax_table:
            for entry in table:
                for expected_entry in expected_entries:
                    found_count += 1 if expected_entry in entry.get_string() else 0
        self.assertEqual(found_count, len(expected_entries))


def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()  # pragma: no cover
