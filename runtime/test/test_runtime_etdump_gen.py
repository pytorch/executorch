# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest

import torch

from executorch.extension.pybindings.test.make_test import (
    create_program,
    ModuleAdd,
)
from executorch.runtime import Runtime, Verification
import os
from executorch.devtools.etdump.serialize import deserialize_from_etdump_flatcc

class RuntimeETDumpGenTest(unittest.TestCase):
    def test_etdump_generation(self):
        """Test etdump generation by creating a program with etdump enabled and verifying the output."""

        ep, inputs = create_program(ModuleAdd())
        runtime = Runtime.get()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the program to a file
            program_path = os.path.join(temp_dir, "test_program.pte")
            with open(program_path, "wb") as f:
                f.write(ep.buffer)

            # Load program with etdump generation enabled
            program = runtime.load_program(
                program_path,
                verification=Verification.Minimal,
                enable_etdump=True,
                debug_buffer_size=int(1e7),  # Large buffer size to ensure all debug info is captured
            )

            # Execute the method
            method = program.load_method("forward")
            outputs = method.execute(inputs)

            # Verify the computation is correct
            self.assertTrue(torch.allclose(outputs[0], inputs[0] + inputs[1]))

            # Write etdump result to files
            etdump_path = os.path.join(temp_dir, "etdump_output.etdp")
            debug_path = os.path.join(temp_dir, "debug_output.bin")
            program.write_etdump_result_to_file(etdump_path, debug_path)

            # Check that files were created
            self.assertTrue(os.path.exists(etdump_path), f"ETDump file not created at {etdump_path}")
            self.assertTrue(os.path.exists(debug_path), f"Debug file not created at {debug_path}")

            # Verify the etdump file is not empty
            etdump_size = os.path.getsize(etdump_path)
            self.assertGreater(etdump_size, 0, "ETDump file is empty")

            # Read and deserialize the etdump file to verify its structure
            with open(etdump_path, "rb") as f:
                etdump_data = f.read()

            # Deserialize the etdump and check its header/structure
            etdump = deserialize_from_etdump_flatcc(etdump_data)

            # Verify ETDump header properties
            self.assertIsInstance(etdump.version, int, "ETDump version should be an integer")
            self.assertGreaterEqual(etdump.version, 0, "ETDump version should be non-negative")

            # Verify run_data structure
            self.assertIsInstance(etdump.run_data, list, "ETDump run_data should be a list")
            self.assertGreater(len(etdump.run_data), 0, "ETDump should contain at least one run data entry")

            # Check the first run_data entry
            run_data = etdump.run_data[0]
            self.assertIsInstance(run_data.events, list, "Run data should contain events list")
            self.assertGreater(len(run_data.events), 0, "Run data should contain at least one events")
