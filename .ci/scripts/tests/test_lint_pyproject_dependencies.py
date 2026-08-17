import importlib.util
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "lint_pyproject_dependencies.py"
spec = importlib.util.spec_from_file_location(
    "lint_pyproject_dependencies", SCRIPT_PATH
)
assert spec is not None
assert spec.loader is not None
lint_pyproject_dependencies = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lint_pyproject_dependencies
spec.loader.exec_module(lint_pyproject_dependencies)
find_direct_reference_dependencies = (
    lint_pyproject_dependencies.find_direct_reference_dependencies
)


class TestLintPyprojectDependencies(unittest.TestCase):
    def write_pyproject(self, content: str) -> Path:
        path = Path(self.tempdir.name) / "pyproject.toml"
        path.write_text(textwrap.dedent(content))
        return path

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_allows_versioned_dependencies(self) -> None:
        path = self.write_pyproject(
            """
            [project]
            dependencies = [
              "torch>=2.12.0a0",
            ]

            [project.optional-dependencies]
            openvino = [
              "openvino>=2025.1.0,<2026.0.0; platform_system == 'Linux'",
            ]
            """
        )

        self.assertEqual(find_direct_reference_dependencies(path), [])

    def test_rejects_project_direct_reference_dependency(self) -> None:
        path = self.write_pyproject(
            """
            [project]
            dependencies = [
              "example @ git+https://github.com/example/example.git@abc123",
            ]
            """
        )

        violations = find_direct_reference_dependencies(path)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].section, "project.dependencies")
        self.assertEqual(
            violations[0].dependency,
            "example @ git+https://github.com/example/example.git@abc123",
        )

    def test_rejects_direct_reference_dependency_with_spaced_extras(self) -> None:
        path = self.write_pyproject(
            """
            [project]
            dependencies = [
              "example [dev] @ git+https://github.com/example/example.git@abc123",
            ]
            """
        )

        violations = find_direct_reference_dependencies(path)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].section, "project.dependencies")
        self.assertEqual(
            violations[0].dependency,
            "example [dev] @ git+https://github.com/example/example.git@abc123",
        )

    def test_rejects_optional_direct_reference_dependency(self) -> None:
        path = self.write_pyproject(
            """
            [project]
            dependencies = []

            [project.optional-dependencies]
            cortex_m = [
              "cmsis_nn @ git+https://github.com/ARM-software/CMSIS-NN.git@abc123",
            ]
            """
        )

        violations = find_direct_reference_dependencies(path)
        self.assertEqual(len(violations), 1)
        self.assertEqual(
            violations[0].section,
            "project.optional-dependencies.cortex_m",
        )
        self.assertEqual(
            violations[0].dependency,
            "cmsis_nn @ git+https://github.com/ARM-software/CMSIS-NN.git@abc123",
        )


if __name__ == "__main__":
    unittest.main()
