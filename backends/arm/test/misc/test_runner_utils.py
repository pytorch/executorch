# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, cast

from executorch.backends.arm.test import runner_utils


class _FakeExecutorchProgramManager:
    def __init__(self, buffer: bytes) -> None:
        self.buffer = buffer

    def exported_program(self):
        return object()


def test_run_corstone_uses_short_input_aliases_in_semihosting_cmd(
    monkeypatch, tmp_path: Path
) -> None:
    long_input_paths = [
        str(tmp_path / ("very_long_input_name_" * 6 + "0.bin")),
        str(tmp_path / ("very_long_input_name_" * 6 + "1.bin")),
    ]

    monkeypatch.setattr(
        runner_utils,
        "save_inputs_to_file",
        lambda exported_program, inputs, intermediate_path: long_input_paths,
    )

    copied_files: list[tuple[str, str]] = []

    def _fake_copyfile(src: str, dst: str) -> None:
        copied_files.append((src, dst))

    monkeypatch.setattr(runner_utils.shutil, "copyfile", _fake_copyfile)

    captured: dict[str, list[str]] = {}

    def _fake_run_cmd(cmd, check=True):
        captured["cmd"] = cmd
        return runner_utils.subprocess.CompletedProcess(
            cmd, 0, stdout=b"OK", stderr=b""
        )

    monkeypatch.setattr(runner_utils, "_run_cmd", _fake_run_cmd)
    monkeypatch.setattr(
        runner_utils,
        "get_output_from_file",
        lambda exported_program, intermediate_path, output_base_name: ("sentinel",),
    )

    elf_path = tmp_path / "arm_executor_runner"
    elf_path.write_bytes(b"")

    output = runner_utils.run_corstone(
        executorch_program_manager=cast(
            Any, _FakeExecutorchProgramManager(buffer=b"pte")
        ),
        inputs=cast(Any, ()),
        intermediate_path=tmp_path,
        target_board="corstone-320",
        elf_path=elf_path,
        timeout=1,
    )

    assert output == ("sentinel",)
    assert [Path(dst).name for _, dst in copied_files] == ["i0.bin", "i1.bin"]

    semihosting_cmd_arg = next(
        arg for arg in captured["cmd"] if "semihosting-cmd_line" in arg
    )
    assert "-i i0.bin" in semihosting_cmd_arg
    assert "-i i1.bin" in semihosting_cmd_arg
    assert long_input_paths[0] not in semihosting_cmd_arg
    assert long_input_paths[1] not in semihosting_cmd_arg
