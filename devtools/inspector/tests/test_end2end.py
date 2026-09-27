import copy
import os
import tempfile
import unittest

import torch
from executorch.devtools.etrecord import generate_etrecord
from executorch.devtools.inspector import Inspector
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)
from torch.export import export


class SimpleModel(torch.nn.Module):
    def forward(self, x, y):
        return x + y


class TestEnd2End(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.etrecord_path = os.path.join(self.temp_dir.name, "etrecord.bin")
        self.etdump_path = os.path.join(self.temp_dir.name, "etdump.etdp")
        self.model = SimpleModel()

    def test_end_to_end_inspector(self):
        # 1. Export the model and generate etrecord
        aten_model = export(
            self.model, (torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32))
        )
        edge_program_manager = to_edge(
            aten_model, compile_config=EdgeCompileConfig(_check_ir_validity=False)
        )
        edge_program_manager_copy = copy.deepcopy(edge_program_manager)
        et_program_manager = edge_program_manager.to_executorch()

        generate_etrecord(
            self.etrecord_path, edge_program_manager_copy, et_program_manager
        )

        # 2. Load the model in C++ runtime and generate etdump
        program = _load_for_executorch_from_buffer(
            et_program_manager.buffer, enable_etdump=True
        )
        program.forward((torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32)))
        program.write_etdump_result_to_file(self.etdump_path)

        # Verify files were created
        self.assertTrue(os.path.exists(self.etrecord_path), "etrecord was not created")
        self.assertTrue(os.path.exists(self.etdump_path), "etdump was not created")

        # 3. Load both into the Inspector API and verify
        inspector = Inspector(etdump_path=self.etdump_path, etrecord=self.etrecord_path)
        df = inspector.to_dataframe()

        # Assertions
        self.assertIsNotNone(df, "Inspector dataframe should not be None")
        self.assertGreater(
            len(df), 0, "Inspector dataframe should have at least one row"
        )

    def tearDown(self):
        self.temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
