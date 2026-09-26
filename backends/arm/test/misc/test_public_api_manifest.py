# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import subprocess  # nosec B404
from pathlib import Path

import executorch.backends.arm as arm
from executorch.backends.arm import LAZY_IMPORTS
from executorch.backends.arm.scripts.public_api_manifest.generate_public_api_manifest import (
    _collect_cmake_public_api,
    _collect_entry,
    _collect_public_api,
    _is_unstable_api,
    _render_manifest,
)
from executorch.exir._warnings import deprecated

REPO_PATH = Path(__file__).resolve().parents[4]
RUNNING_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "public_api_manifests"
    / "api_manifest_running.toml"
)


def _entry_block(path: str, entry: dict[str, str]) -> str:
    return "\n".join(
        [
            f"[python.{path}]",
            f'kind = "{entry["kind"]}"',
            f'signature = "{entry["signature"]}"',
        ]
    )


def test_public_api_manifest_entries_are_well_formed():
    entries = _collect_public_api()
    expected_top_level_entries = {
        name for name in LAZY_IMPORTS if not _is_unstable_api(getattr(arm, name))
    }

    assert entries
    assert {path.split(".")[0] for path in entries} == expected_top_level_entries

    for path, entry in entries.items():
        assert entry["kind"] in {"class", "enum", "function"}
        assert entry["signature"].startswith(path)
        assert "(" in entry["signature"]
        assert not path.endswith(".__init__")

        if "." in path:
            assert path.rsplit(".", 1)[0] in entries


def test_public_api_manifest_matches_generator():
    entries = _collect_public_api()
    cmake_entries = _collect_cmake_public_api(REPO_PATH)
    manifest = _render_manifest(entries, cmake_entries)

    assert manifest == _render_manifest(entries, cmake_entries)
    assert manifest.startswith("# Copyright ")
    assert "[python]" in manifest
    assert "[cmake]" in manifest

    for path, entry in entries.items():
        assert _entry_block(path, entry) in manifest

    for path, entry in cmake_entries.items():
        assert entry["kind"] in {"function", "macro"}
        assert entry["signature"].startswith(f"{path}(")
        assert (
            "\n".join(
                [
                    f"[cmake.{path}]",
                    f'kind = "{entry["kind"]}"',
                    f'signature = "{entry["signature"]}"',
                ]
            )
            in manifest
        )

    assert manifest == Path(RUNNING_MANIFEST_PATH).read_text(encoding="utf-8")


def test_cmake_public_api_uses_explicit_repo_path(tmp_path):
    facade = tmp_path / "backends" / "arm" / "cmake" / "ArmRunnerUtils.cmake"
    facade.parent.mkdir(parents=True)
    facade.write_text(
        'message("ARM_RUNNER_UTILS_API|arm_runner_foo|function|arm_runner_foo()")',
        encoding="utf-8",
    )

    assert _collect_cmake_public_api(tmp_path) == {
        "arm_runner_foo": {
            "kind": "function",
            "signature": "arm_runner_foo()",
        }
    }


def test_cmake_facade_ignores_consumer_commands(tmp_path):
    cmake = shutil.which("cmake")
    assert cmake is not None
    facade = REPO_PATH / "backends" / "arm" / "cmake" / "ArmRunnerUtils.cmake"
    script = tmp_path / "consumer.cmake"
    script.write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "function(arm_runner_consumer_helper)\n"
        "endfunction()\n"
        'include("{}")\n'.format(facade),
        encoding="utf-8",
    )

    result = subprocess.run(  # nosec B603
        [cmake, "-P", str(script)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_public_api_manifest_collection_handles_deprecated_symbols():
    @deprecated("old foo")
    def old_foo(x: int) -> int:
        return x

    old_foo.__module__ = "executorch.backends.arm.synthetic"
    entries: dict[str, dict[str, str]] = {}

    _collect_entry("old_foo", old_foo, entries)

    assert "old_foo" not in entries


def test_public_api_manifest_collection_can_include_deprecated_symbols():
    @deprecated("old foo")
    def old_foo(x: int) -> int:
        return x

    old_foo.__module__ = "executorch.backends.arm.synthetic"
    entries: dict[str, dict[str, str]] = {}

    _collect_entry("old_foo", old_foo, entries, include_deprecated=True)

    assert entries["old_foo"]["kind"] == "function"
    assert entries["old_foo"]["signature"] == "old_foo(x: int) -> int"


def test_public_api_manifest_collection_excludes_init_for_equivalent_classes():
    class ExplicitInit:
        def __init__(self, x: int = 0) -> None:
            del x

        def method(self) -> int:
            return 1

    class ImplicitInit:
        def method(self) -> int:
            return 2

    ExplicitInit.__module__ = "executorch.backends.arm.synthetic"
    ImplicitInit.__module__ = "executorch.backends.arm.synthetic"
    entries: dict[str, dict[str, str]] = {}

    _collect_entry("ExplicitInit", ExplicitInit, entries)
    _collect_entry("ImplicitInit", ImplicitInit, entries)

    assert "ExplicitInit.__init__" not in entries
    assert "ImplicitInit.__init__" not in entries
    assert entries["ExplicitInit.method"]["kind"] == "function"
    assert entries["ImplicitInit.method"]["kind"] == "function"
