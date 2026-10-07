# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Tests for the torch requirement a release wheel declares.
#
# 1.5.0 and 1.5.1 shipped with no torch requirement at all, and only a release build
# exercises it, so main's own wheel jobs, which are nightlies, never would.

import ast
import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[3]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INSTALL_UTILS = _load_module("install_utils", ROOT / "install_utils.py")


class TestReleaseTorchRequirement(unittest.TestCase):
    def requirement(self, build_version, torch_version="2.14.1+cu132"):
        with mock.patch.object(
            INSTALL_UTILS.importlib.metadata, "version", return_value=torch_version
        ) as version:
            requirement = INSTALL_UTILS.release_torch_requirement(build_version)
        if requirement is not None:
            version.assert_called_once_with("torch")
        return requirement

    def test_release_and_candidate_builds_declare_the_built_minor(self):
        # Linux and Windows carry the build variant after a plus sign, macOS carries none. A post
        # release is a hotfix of a release, so it ships against the same torch.
        for build_version in ("1.6.0+cpu", "1.6.0+cu132", "1.6.0", "1.6.0.post1+cpu"):
            with self.subTest(build_version=build_version):
                self.assertEqual(
                    self.requirement(build_version), "torch>=2.14.0a0,<2.15"
                )

    def test_nightly_and_local_builds_declare_nothing(self):
        for build_version in (
            "1.6.0.dev20261001+cpu",
            "1.6.0.dev20261001",
            "1.6.0rc1",
            "1.6.0.post1.dev2",
            "",
            None,
        ):
            with self.subTest(build_version=build_version):
                self.assertIsNone(self.requirement(build_version))

    def test_a_release_without_torch_names_build_version(self):
        with mock.patch.object(
            INSTALL_UTILS.importlib.metadata,
            "version",
            side_effect=INSTALL_UTILS.importlib.metadata.PackageNotFoundError("torch"),
        ):
            with self.assertRaisesRegex(RuntimeError, "BUILD_VERSION=1.6.0"):
                INSTALL_UTILS.release_torch_requirement("1.6.0")

    def test_the_range_follows_the_installed_torch(self):
        self.assertEqual(
            self.requirement("1.7.0+cpu", "2.15.0"), "torch>=2.15.0a0,<2.16"
        )

    def test_the_range_admits_the_torch_the_wheel_was_built_on(self):
        # A nightly snapshot sorts below a0, which is how a CUDA row built on one would have
        # declared a range excluding its own torch.
        for torch_version in (
            "2.14.0",
            "2.14.1+cu130",
            "2.14.0a0+git0123abc",
            "2.14.0.dev20260810+cu134",
        ):
            with self.subTest(torch_version=torch_version):
                requirement = Requirement(self.requirement("1.6.0+cpu", torch_version))
                self.assertTrue(
                    requirement.specifier.contains(torch_version, prereleases=True)
                )
                self.assertFalse(requirement.specifier.contains("2.13.1"))
                self.assertFalse(
                    requirement.specifier.contains(
                        "2.15.0.dev20261001", prereleases=True
                    )
                )


class TestSetupDeclaresIt(unittest.TestCase):
    def torch_dependencies(self, build_version):
        # Runs the real _torch_dependencies from setup.py, which cannot be imported because
        # importing it runs setup().
        path = ROOT / "setup.py"
        function = next(
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "_torch_dependencies"
        )
        namespace = {"List": list, "install_utils": INSTALL_UTILS, "os": os}
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        environment = {} if build_version is None else {"BUILD_VERSION": build_version}
        with mock.patch.dict(os.environ, environment, clear=True):
            with mock.patch.object(
                INSTALL_UTILS.importlib.metadata, "version", return_value="2.14.1"
            ):
                return namespace["_torch_dependencies"]()

    def test_setup_reads_build_version(self):
        self.assertEqual(
            self.torch_dependencies("1.6.0+cpu"), ["torch>=2.14.0a0,<2.15"]
        )
        for build_version in ("1.6.0.dev20261001+cpu", None):
            with self.subTest(build_version=build_version):
                self.assertEqual(self.torch_dependencies(build_version), [])

    def test_only_the_full_wheel_declares_it(self):
        # setup.py is read rather than imported, because importing it runs setup().
        module = ast.parse((ROOT / "setup.py").read_text())
        called = [
            {
                call.func.id
                for call in ast.walk(node.value)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            }
            for node in ast.walk(module)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Subscript)
                and getattr(target.slice, "value", None) == "install_requires"
                for target in node.targets
            )
        ]
        full = [names for names in called if "_base_dependencies" in names]
        minimal = [names for names in called if "_minimal_dependencies" in names]
        self.assertEqual((len(full), len(minimal)), (1, 1), called)
        self.assertIn("_torch_dependencies", full[0])
        self.assertNotIn("_torch_dependencies", minimal[0])


if __name__ == "__main__":
    unittest.main()
