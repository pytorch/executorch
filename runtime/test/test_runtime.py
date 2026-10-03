# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from executorch.extension.pybindings.test.make_test import (
    create_program,
    ModuleAdd,
    ModuleMulti,
)
from executorch.exir import ExecutorchBackendConfig, to_edge
from executorch.exir.passes import MemoryPlanningPass
from executorch.runtime import Runtime, Verification


class ModuleCacheUpdate(torch.nn.Module):
    """Writes one row of a cache passed in as an input, like a static KV cache."""

    def forward(self, x, cache, pos):
        cache.index_copy_(0, pos, x)
        return cache.sum(0, keepdim=True)


def _load_unplanned_io_method(module, inputs):
    """Loads `forward` exported with neither its inputs nor its outputs memory
    planned. Returns the program buffer too: the loaded program reads it in
    place, so it must outlive the method."""
    config = ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=False, alloc_graph_output=False
        )
    )
    buffer = to_edge(torch.export.export(module, inputs)).to_executorch(config).buffer
    runtime = Runtime.get()
    method = runtime.load_program(
        buffer, verification=Verification.Minimal
    ).load_method("forward")
    return method, buffer


class RuntimeTest(unittest.TestCase):
    def test_runtime_import_does_not_load_the_exporter(self):
        # A fresh interpreter, since this process has already imported exir.
        probe = (
            "import sys; import executorch.runtime; "
            "print(sorted(m for m in ('executorch.exir', 'torch._dynamo') if m in sys.modules))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        self.assertEqual(result.stdout.strip(), "[]")

    def test_smoke(self):
        ep, inputs = create_program(ModuleAdd())
        runtime = Runtime.get()
        # Demonstrate that get() returns a singleton.
        runtime2 = Runtime.get()
        self.assertTrue(runtime is runtime2)
        program = runtime.load_program(ep.buffer, verification=Verification.Minimal)
        method = program.load_method("forward")
        outputs = method.execute(inputs)
        self.assertTrue(torch.allclose(outputs[0], inputs[0] + inputs[1]))

    def test_mutated_input_is_updated_in_place(self):
        # An unplanned input the method mutates is written where it lives, and
        # its write-back output is that very tensor rather than a copy.
        x, cache, pos = torch.ones(1, 4), torch.zeros(3, 4), torch.tensor([1])
        method, _buffer = _load_unplanned_io_method(ModuleCacheUpdate(), (x, cache, pos))
        for step, row in enumerate((1, 2)):
            outputs = method.execute((x * (step + 1), cache, torch.tensor([row])))
            self.assertIs(outputs[0], cache)
        expected = torch.zeros(3, 4)
        expected[1], expected[2] = 1.0, 2.0
        self.assertTrue(torch.equal(cache, expected))
        self.assertTrue(torch.equal(outputs[1], expected.sum(0, keepdim=True)))

    def test_set_output_binds_caller_storage(self):
        x, cache, pos = torch.ones(1, 4), torch.zeros(3, 4), torch.tensor([1])
        method, _buffer = _load_unplanned_io_method(ModuleCacheUpdate(), (x, cache, pos))
        # Outputs are [the mutated cache, the sum]: bind the sum to a caller
        # tensor, which the method then writes and returns uncloned.
        summed = torch.empty(1, 4)
        method.set_output(summed, 1)
        for row in (1, 2):
            outputs = method.execute((x, cache, torch.tensor([row])))
            self.assertIs(outputs[1], summed)
            self.assertTrue(torch.equal(summed, cache.sum(0, keepdim=True)))

    def test_set_output_rejects_bad_bindings(self):
        x, cache, pos = torch.ones(1, 4), torch.zeros(3, 4), torch.tensor([1])
        method, _buffer = _load_unplanned_io_method(ModuleCacheUpdate(), (x, cache, pos))
        with self.assertRaises(IndexError):
            method.set_output(torch.empty(1, 4), 2)
        with self.assertRaises(RuntimeError):
            method.set_output(torch.zeros(4, 3).t(), 0)
        with self.assertRaises(RuntimeError):
            method.set_output(torch.empty(1, 2), 1)

    def test_module_with_multiple_method_names(self):
        ep, inputs = create_program(ModuleMulti())
        runtime = Runtime.get()

        program = runtime.load_program(ep.buffer, verification=Verification.Minimal)
        self.assertEqual(program.method_names, set({"forward", "forward2"}))
        method = program.load_method("forward")
        outputs = method.execute(inputs)
        self.assertTrue(torch.allclose(outputs[0], inputs[0] + inputs[1]))

        method = program.load_method("forward2")
        outputs = method.execute(inputs)
        self.assertTrue(torch.allclose(outputs[0], inputs[0] + inputs[1] + 1))

    def test_print_operator_names(self):
        ep, inputs = create_program(ModuleAdd())
        runtime = Runtime.get()

        operator_names = runtime.operator_registry.operator_names
        self.assertGreater(len(operator_names), 0)

        self.assertIn("aten::add.out", operator_names)

    def test_load_program_with_path(self):
        ep, inputs = create_program(ModuleAdd())
        runtime = Runtime.get()

        def test_add(program):
            method = program.load_method("forward")
            outputs = method.execute(inputs)
            self.assertTrue(torch.allclose(outputs[0], inputs[0] + inputs[1]))

        with tempfile.NamedTemporaryFile() as f:
            f.write(ep.buffer)
            f.flush()
            # filename
            program = runtime.load_program(f.name)
            test_add(program)
            # pathlib.Path
            path = Path(f.name)
            program = runtime.load_program(path)
            test_add(program)
            # BytesIO
            with open(f.name, "rb") as f:
                program = runtime.load_program(f.read())
                test_add(program)

    def test_load_program_with_file_like_objects(self):
        """Regression test: Ensure file-like objects (BytesIO, etc.) work correctly.

        Previously, isinstance(data, BinaryIO) check didn't work because BinaryIO
        is a typing protocol. Fixed by using hasattr(data, 'read') duck-typing.
        """
        ep, inputs = create_program(ModuleAdd())
        runtime = Runtime.get()

        def test_add(program):
            method = program.load_method("forward")
            outputs = method.execute(inputs)
            self.assertTrue(torch.allclose(outputs[0], inputs[0] + inputs[1]))

        # Test with BytesIO
        bytesio = io.BytesIO(ep.buffer)
        program = runtime.load_program(bytesio)
        test_add(program)

        # Test with bytes
        program = runtime.load_program(bytes(ep.buffer))
        test_add(program)

        # Test with bytearray
        program = runtime.load_program(bytearray(ep.buffer))
        test_add(program)
