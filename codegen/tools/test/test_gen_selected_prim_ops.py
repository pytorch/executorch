# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path

import yaml

from executorch.codegen.tools.gen_selected_prim_ops import (
    normalize_op_name,
    write_selected_prim_ops,
)


class TestGenSelectedPrimOps(unittest.TestCase):
    def test_normalizes_aten_op_with_leading_underscore(self) -> None:
        self.assertEqual(
            normalize_op_name("aten::_local_scalar_dense"),
            "INCLUDE_ATEN_LOCAL_SCALAR_DENSE",
        )

    def test_writes_selected_prim_ops_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "selected_operators.yaml"
            yaml_path.write_text(
                yaml.safe_dump(
                    {
                        "operators": {
                            "executorch_prim::et_view.default": {},
                            "aten::_local_scalar_dense": {},
                        },
                        "et_kernel_metadata": {
                            "aten::sym_size.int": ["default"],
                        },
                    }
                )
            )

            from executorch.codegen.tools.gen_selected_prim_ops import main

            main(
                [
                    f"--op-selection-yaml-path={yaml_path}",
                    f"--output-dir={temp_dir}",
                ]
            )

            header = (Path(temp_dir) / "selected_prim_ops.h").read_text()
            self.assertIn("#define INCLUDE_ATEN_LOCAL_SCALAR_DENSE", header)
            self.assertIn("#define INCLUDE_ATEN_SYM_SIZE_INT", header)
            self.assertIn("#define INCLUDE_EXECUTORCH_PRIM_ET_VIEW_DEFAULT", header)

    def test_writes_empty_header_for_empty_op_list(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            write_selected_prim_ops([], temp_dir)

            header = (Path(temp_dir) / "selected_prim_ops.h").read_text()
            self.assertNotIn("#define INCLUDE_", header)


if __name__ == "__main__":
    unittest.main()
