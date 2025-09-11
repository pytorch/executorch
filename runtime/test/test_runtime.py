# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path

import torch

from executorch.extension.pybindings.test.make_test import (
    create_program,
    ModuleAdd,
    ModuleMulti,
)
from executorch.runtime import Runtime, Verification


class RuntimeTest(unittest.TestCase):
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
