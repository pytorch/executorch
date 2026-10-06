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
from executorch.exir import ExecutorchBackendConfig, to_edge
from executorch.exir.passes import MemoryPlanningPass

from executorch.extension.pybindings.test.make_test import (
    create_program,
    ModuleAdd,
    ModuleMulti,
)
from executorch.runtime import Runtime, Verification


class ModuleCacheUpdate(torch.nn.Module):
    """Writes one row of a cache passed in as an input, like a static KV cache."""

    def forward(self, x, cache, pos):
        cache.index_copy_(0, pos, x)
        return cache.sum(0, keepdim=True)


class ModuleRecurrent(torch.nn.Module):
    def forward(self, x, state):
        return state + x


class ModuleDouble(torch.nn.Module):
    def forward(self, x):
        return x * 2


class ModuleDoubleChannelsLast(torch.nn.Module):
    def forward(self, x):
        return (x * 2).contiguous(memory_format=torch.channels_last)


class ModuleDoubleTwice(torch.nn.Module):
    def forward(self, x):
        y = x * 2
        return y, y


def _cache_inputs():
    return torch.ones(1, 4), torch.zeros(3, 4), torch.tensor([1])


def _load_unplanned_io_method(
    module, inputs, dynamic_shapes=None, plan_inputs=False, plan_outputs=False
):
    """Loads `forward` exported with its inputs and outputs not memory planned,
    unless asked. Returns the program buffer too: the loaded program reads it
    in place, so it must outlive the method."""
    config = ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=plan_inputs, alloc_graph_output=plan_outputs
        )
    )
    exported = torch.export.export(module, inputs, dynamic_shapes=dynamic_shapes)
    buffer = to_edge(exported).to_executorch(config).buffer
    program = Runtime.get().load_program(buffer, verification=Verification.Minimal)
    return program.load_method("forward"), buffer


def _load_cache_update_method():
    return _load_unplanned_io_method(ModuleCacheUpdate(), _cache_inputs())


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
        x, cache, _ = _cache_inputs()
        method, _buffer = _load_cache_update_method()
        for step, row in enumerate((1, 2)):
            outputs = method.execute((x * (step + 1), cache, torch.tensor([row])))
            self.assertIs(outputs[0], cache)
        expected = torch.zeros(3, 4)
        expected[1], expected[2] = 1.0, 2.0
        self.assertTrue(torch.equal(cache, expected))
        self.assertTrue(torch.equal(outputs[1], expected.sum(0, keepdim=True)))

    def test_mutated_planned_input_is_not_updated(self):
        x, cache, pos = _cache_inputs()
        method, _buffer = _load_unplanned_io_method(
            ModuleCacheUpdate(), _cache_inputs(), plan_inputs=True
        )
        outputs = method.execute((x, cache, pos))
        self.assertIsNot(outputs[0], cache)
        self.assertTrue(torch.equal(cache, torch.zeros(3, 4)))
        self.assertTrue(torch.equal(outputs[0][1], torch.ones(4)))

    def test_output_fed_back_as_input_is_not_that_input(self):
        # An uncloned output fed back in shares the input's memory, but is not
        # an input the method mutates.
        x, state = torch.ones(2), torch.zeros(2)
        method, _buffer = _load_unplanned_io_method(ModuleRecurrent(), (x, state))
        for step in (1, 2):
            fed = state
            state = method._method((x, fed), clone_outputs=False)[0]
            self.assertIsNot(state, fed)
            self.assertTrue(torch.equal(state, torch.full((2,), float(step))))
        fresh = torch.zeros(2)
        outputs = method._method((x, fresh), clone_outputs=False)
        self.assertIsNot(outputs[0], fresh)
        self.assertTrue(torch.equal(fresh, torch.zeros(2)))
        self.assertTrue(torch.equal(outputs[0], torch.ones(2)))

    def test_set_output_binds_caller_storage(self):
        x, cache, _ = _cache_inputs()
        method, _buffer = _load_cache_update_method()
        # Outputs are [the mutated cache, the sum]: bind the sum to a caller
        # tensor, which the method then writes and returns uncloned.
        summed = torch.empty(1, 4)
        method.set_output(summed, 1)
        for row in (1, 2):
            outputs = method.execute((x, cache, torch.tensor([row])))
            self.assertIs(outputs[1], summed)
            self.assertTrue(torch.equal(summed, cache.sum(0, keepdim=True)))

    def test_set_output_rebinds(self):
        x, cache, pos = _cache_inputs()
        method, _buffer = _load_cache_update_method()
        first, second = torch.zeros(1, 4), torch.zeros(1, 4)
        method.set_output(first, 1)
        method.set_output(second, 1)
        outputs = method.execute((x, cache, pos))
        self.assertIs(outputs[1], second)
        self.assertTrue(torch.equal(second, torch.ones(1, 4)))
        self.assertTrue(torch.equal(first, torch.zeros(1, 4)))

    def test_execute_refused_after_failed_set_inputs(self):
        x, cache, pos = _cache_inputs()
        method, _buffer = _load_cache_update_method()
        method._method.set_inputs((x, cache, pos))
        # Fails on the position, after the new cache was already set.
        new_cache = torch.zeros(3, 4)
        with self.assertRaises(RuntimeError):
            method._method.set_inputs((x, new_cache, torch.tensor([1.0])))
        with self.assertRaises(RuntimeError):
            method._method.execute()
        outputs = method.execute((x, new_cache, pos))
        self.assertIs(outputs[0], new_cache)
        self.assertTrue(torch.equal(cache, torch.zeros(3, 4)))

    def test_caller_backed_outputs_are_not_cloned(self):
        x, cache, pos = _cache_inputs()
        method, _buffer = _load_cache_update_method()
        method.set_output(torch.empty(1, 4), 1)
        with torch.profiler.profile() as profile:
            method.execute((x, cache, pos))
        clones = [e for e in profile.events() if e.name == "aten::clone"]
        self.assertEqual(clones, [])

    def test_set_output_keeps_earlier_outputs_alive(self):
        x, cache, pos = _cache_inputs()
        method, _buffer = _load_cache_update_method()
        earlier = method._method((x, cache, pos), clone_outputs=False)[1]
        expected = earlier.clone()
        method.set_output(torch.empty(1, 4), 1)
        method.execute((x * 2, cache, torch.tensor([2])))
        self.assertTrue(torch.equal(earlier, expected))

    def test_set_output_narrows_dynamic_output(self):
        dynamic_shapes = ({0: torch.export.Dim("n", max=8)},)
        method, _buffer = _load_unplanned_io_method(
            ModuleDouble(), (torch.ones(4, 3),), dynamic_shapes
        )
        bound = torch.empty(8, 3)
        method.set_output(bound, 0)
        x = torch.arange(6.0).reshape(2, 3)
        output = method.execute((x,))[0]
        self.assertEqual(output.shape, (2, 3))
        self.assertEqual(output.data_ptr(), bound.data_ptr())
        self.assertTrue(torch.equal(output, x * 2))

    def test_set_output_rejects_bad_bindings(self):
        method, _buffer = _load_cache_update_method()
        bad_bindings = {
            "out of range": (IndexError, torch.empty(1, 4), 2),
            "non contiguous": (RuntimeError, torch.empty(1, 8)[:, ::2], 1),
            "too small": (RuntimeError, torch.empty(1, 2), 1),
            "wrong dtype": (RuntimeError, torch.empty(1, 4, dtype=torch.float64), 1),
            "wrong device": (RuntimeError, torch.empty(1, 4, device="meta"), 1),
        }
        for name, (error, tensor, index) in bad_bindings.items():
            with self.subTest(name), self.assertRaises(error):
                method.set_output(tensor, index)

    def test_set_output_rejects_outputs_it_cannot_own(self):
        x = torch.ones(2)
        method, _buffer = _load_unplanned_io_method(
            ModuleDouble(), (x,), plan_outputs=True
        )
        with self.subTest("memory planned"), self.assertRaises(RuntimeError):
            method.set_output(torch.empty(2), 0)
        method, _buffer = _load_unplanned_io_method(ModuleDoubleTwice(), (x,))
        with self.subTest("returned twice"), self.assertRaises(RuntimeError):
            method.set_output(torch.empty(2), 0)
        image = torch.ones(1, 2, 3, 4)
        method, _buffer = _load_unplanned_io_method(
            ModuleDoubleChannelsLast(), (image,)
        )
        with self.subTest("channels last"), self.assertRaises(RuntimeError):
            method.set_output(torch.empty(1, 2, 3, 4), 0)

    def test_set_output_rejects_mutated_input(self):
        x, cache, pos = _cache_inputs()
        method, _buffer = _load_cache_update_method()
        method.execute((x, cache, pos))
        with self.assertRaises(RuntimeError):
            method.set_output(torch.zeros(3, 4), 0)
        # Bound before any inputs are set, the first call removes the binding
        # and raises, and later calls work.
        method, _buffer = _load_cache_update_method()
        method.set_output(torch.zeros(3, 4), 0)
        with self.assertRaises(RuntimeError):
            method.execute((x, cache, pos))
        outputs = method.execute((x, cache, pos))
        self.assertIs(outputs[0], cache)
        self.assertTrue(torch.equal(cache[1], torch.ones(4)))

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
