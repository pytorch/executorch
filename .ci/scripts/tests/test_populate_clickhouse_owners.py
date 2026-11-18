# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "populate_clickhouse_owners", ROOT / ".ci/scripts/populate_clickhouse_owners.py"
)
assert SPEC is not None and SPEC.loader is not None
owners = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(owners)

CODEOWNERS = """\
# Comment
/backends/apple @f
/backends/arm @a
/backends/xnnpack @b
/examples/arm @a
/exir @c
/exir/backend @d
/backends/test
/profiler @g
CMakeLists.txt @e
"""


class PopulateClickhouseOwnersTest(unittest.TestCase):
    def owner(self, path, source=""):
        return owners.owner_of(path, source, owners.codeowners_areas(CODEOWNERS))

    def test_backends_examples_and_extensions_use_their_name(self):
        self.assertEqual(self.owner("backends/arm/test/test_z.py"), ["arm"])
        self.assertEqual(self.owner("backends/xnnpack/test/test_x.py"), ["xnnpack"])
        self.assertEqual(self.owner("examples/arm/test_example.py"), ["arm"])
        self.assertEqual(self.owner("extension/llm/test/test_runner.py"), ["llm"])

    def test_apple_splits_into_its_backends(self):
        self.assertEqual(self.owner("backends/apple/coreml/test/test_a.py"), ["coreml"])
        self.assertEqual(self.owner("backends/apple/metal/tests/test_b.py"), ["metal"])
        self.assertEqual(self.owner("backends/apple/test_c.py"), ["apple"])

    def test_nested_areas_use_their_top_level_name(self):
        self.assertEqual(self.owner("exir/backend/test/test_x.py"), ["exir"])
        self.assertEqual(self.owner("exir/tests/test_y.py"), ["exir"])

    def test_aliases(self):
        self.assertEqual(
            self.owner("backends/test/suite/test_ops.py"), ["backend_tests"]
        )
        self.assertEqual(self.owner("profiler/test/test_p.py"), ["devtools"])
        self.assertEqual(self.owner(".ci/scripts/tests/test_d.py"), ["ci"])
        self.assertEqual(self.owner("test/end2end/test_e.py"), ["core_tests"])

    def test_unanchored_patterns_match_at_any_depth(self):
        self.assertEqual(self.owner("tools/CMakeLists.txt"), ["build"])

    def test_unmatched_files_fall_back_to_their_directory_name(self):
        self.assertEqual(self.owner("backends/cuda/tests/test_a.py"), ["cuda"])
        self.assertEqual(self.owner("examples/models/test_b.py"), ["models"])
        self.assertEqual(self.owner("kernels/test/test_c.py"), ["kernels"])

    def test_a_module_header_overrides_codeowners(self):
        source = '# Owner(s): ["module: quantization", "oncall: x"]\nimport os\n'
        self.assertEqual(self.owner("exir/tests/test_q.py", source), ["quantization"])

    def test_only_python_test_files_outside_excluded_directories_are_collected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "CODEOWNERS").write_text(CODEOWNERS)
            paths = [
                "exir/tests/test_a.py",
                "exir/tests/b_test.py",
                "exir/tests/helpers.py",
                "exir/fb/test_internal.py",
                "third-party/x/test_vendored.py",
                "exir/tests/test_data.json",
            ]
            for path in paths:
                (repo / path).parent.mkdir(parents=True, exist_ok=True)
                (repo / path).write_text("")
            self.assertEqual(
                owners.collect_test_owners(repo, paths),
                {"exir/tests/test_a.py": ["exir"], "exir/tests/b_test.py": ["exir"]},
            )

    def test_no_test_files_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "CODEOWNERS").write_text(CODEOWNERS)
            with self.assertRaises(ValueError):
                owners.collect_test_owners(Path(tmp), ["README.md"])

    def test_updates_cover_new_changed_and_deleted_files_only(self):
        current = {"a.py": ["x"], "b.py": ["y", "z"], "c.py": ["w"]}
        stored = {"b.py": ["z", "y"], "c.py": ["v"], "d.py": ["u"]}
        self.assertEqual(
            owners.get_owner_updates("pytorch/executorch", current, stored),
            [
                ("pytorch/executorch", "a.py", ["x"]),
                ("pytorch/executorch", "c.py", ["w"]),
                ("pytorch/executorch", "d.py", []),
            ],
        )


if __name__ == "__main__":
    unittest.main()
