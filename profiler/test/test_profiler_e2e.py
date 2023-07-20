import unittest

import torch

# torch.ops.load_library("//executorch/kernels/portable:custom_ops_generated_lib")

from executorch import exir
from executorch.exir.serialize import serialize_to_flatbuffer

# pyre-ignore: Undefined import [21]: Could not find a module corresponding to import `executorch.extension.pybindings.portable`.
from executorch.extension.pybindings.portable import (
    _create_profile_block,
    _dump_profile_results,
    _load_for_executorch_from_buffer,
    _reset_profile_results,
)
from executorch.profiler.parse_profiler_results import (
    deserialize_profile_results,
    profile_aggregate_framework_tax,
    profile_framework_tax_table,
    profile_table,
)
from executorch.pytree import tree_flatten


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = 3 * torch.ones(2, 2, dtype=torch.float)
        self.b = 2 * torch.ones(2, 2, dtype=torch.float)

    def forward(self, x):
        a = torch.mul(self.a, x)
        b = torch.add(a, self.b)
        return b


class TestCustomOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        model = Module()
        inputs = (torch.ones(2, 2, dtype=torch.float),)
        program = exir.capture(model, inputs).to_edge().to_executorch().program
        cls.flatbuff_without_stacktrace = serialize_to_flatbuffer(program)
        # pyre-ignore: Undefined attribute [16]: Module `executorch.extension.pybindings` has no attribute `portable`.
        cls.module = _load_for_executorch_from_buffer(cls.flatbuff_without_stacktrace)

        # pyre-fixme[16]: Module `pytree` has no attribute `tree_flatten`.
        cls.inputs_flattened, _ = tree_flatten(inputs)
        cls.module.run_method("forward", tuple(cls.inputs_flattened))
        # pyre-ignore: Undefined attribute [16]: Module `executorch.extension.pybindings` has no attribute `portable`.
        prof_dump = _dump_profile_results()
        cls.prof_results, cls.mem_results = deserialize_profile_results(prof_dump)
        cls.expect_ops = ["native_call_add.out", "native_call_mul.out"]

    def test_profiler_new_block(self) -> None:
        block_names = ["block_1", "block_2"]
        # pyre-ignore: Undefined attribute [16]: Module `executorch.extension.pybindings` has no attribute `portable`.
        _reset_profile_results()
        # pyre-ignore: Undefined attribute [16]: Module `executorch.extension.pybindings` has no attribute `portable`.
        _create_profile_block(block_names[0])
        self.module.run_method("forward", tuple(self.inputs_flattened))
        # pyre-ignore: Undefined attribute [16]: Module `executorch.extension.pybindings` has no attribute `portable`.
        _create_profile_block(block_names[1])
        self.module.run_method("forward", tuple(self.inputs_flattened))
        # pyre-ignore: Undefined attribute [16]: Module `executorch.extension.pybindings` has no attribute `portable`.
        prof_dump = _dump_profile_results()
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
            self.assertTrue(len(framework_tax.kernel_time) == 1)
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


if __name__ == "__main__":
    unittest.main()
