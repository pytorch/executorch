# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
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


def test_get_elf_path_uses_repo_root_candidates(monkeypatch, tmp_path: Path) -> None:
    elf_path = (
        tmp_path
        / "arm_test"
        / "arm_semihosting_executor_runner_corstone-300"
        / "arm_executor_runner"
    )
    elf_path.parent.mkdir(parents=True)
    elf_path.write_bytes(b"")

    monkeypatch.setattr(runner_utils, "_elf_search_roots", lambda: [tmp_path])
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    assert runner_utils.get_elf_path("corstone-300") == str(elf_path)


def test_get_elf_path_accepts_nested_runner_output(monkeypatch, tmp_path: Path) -> None:
    elf_path = (
        tmp_path
        / "arm_test"
        / "arm_semihosting_executor_runner_corstone-300"
        / "examples"
        / "arm"
        / "executor_runner"
        / "arm_executor_runner"
    )
    elf_path.parent.mkdir(parents=True)
    elf_path.write_bytes(b"")

    monkeypatch.setattr(runner_utils, "_elf_search_roots", lambda: [tmp_path])

    assert runner_utils.get_elf_path("corstone-300") == str(elf_path)


def test_shape_inference_json_uses_tosa_input_layout(tmp_path: Path) -> None:
    test_case_path = tmp_path / "test_case.json"
    artifact_path = tmp_path / "model.tosa"
    input_tensor = torch.randn(1, 3, 4, 5).to(memory_format=torch.channels_last)

    runner_utils.TosaReferenceModelDispatch()._generate_shape_inference_json(
        b"",
        artifact_path,
        test_case_path,
        ["input"],
        (input_tensor,),
    )

    test_case = json.loads(test_case_path.read_text(encoding="utf-8"))

    assert test_case == {
        "tosa_file": str(artifact_path),
        "shapes": {"input": [1, 4, 5, 3]},
    }


def test_numpy_to_torch_tensor_converts_dynamic_nhwc_output(monkeypatch) -> None:
    symbolic_dim = object()
    output_tensor = SimpleNamespace(
        shape=(1, 3, symbolic_dim, 5),
        dtype=torch.float32,
        dim_order=lambda: runner_utils.NHWC_ORDER,
    )
    monkeypatch.setattr(
        runner_utils, "get_first_fake_tensor", lambda output_node: output_tensor
    )
    array = np.arange(60, dtype=np.float32).reshape(1, 4, 5, 3)

    result = runner_utils.numpy_to_torch_tensor(array, cast(Any, object()))

    assert result.shape == (1, 3, 4, 5)
    assert result.is_contiguous()
    torch.testing.assert_close(result, torch.from_numpy(array).permute(0, 3, 1, 2))


def test_numpy_to_torch_tensor_converts_concrete_nhwc_output_to_contiguous(
    monkeypatch,
) -> None:
    output_tensor = SimpleNamespace(
        shape=(1, 3, 4, 5),
        dtype=torch.int8,
        dim_order=lambda: runner_utils.NHWC_ORDER,
    )
    monkeypatch.setattr(
        runner_utils, "get_first_fake_tensor", lambda output_node: output_tensor
    )
    array = np.arange(60, dtype=np.int8).reshape(1, 4, 5, 3)

    result = runner_utils.numpy_to_torch_tensor(array, cast(Any, object()))
    expected = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()

    assert result.is_contiguous()
    torch.testing.assert_close(result, expected)


def test_numpy_to_torch_tensor_converts_dynamic_nnhwc_output(monkeypatch) -> None:
    symbolic_dim = object()
    output_tensor = SimpleNamespace(
        shape=(1, 2, 3, symbolic_dim, 5),
        dtype=torch.float32,
        dim_order=lambda: runner_utils.NNHWC_ORDER,
    )
    monkeypatch.setattr(
        runner_utils, "get_first_fake_tensor", lambda output_node: output_tensor
    )
    array = np.arange(120, dtype=np.float32).reshape(1, 2, 4, 5, 3)

    result = runner_utils.numpy_to_torch_tensor(array, cast(Any, object()))

    assert result.shape == (1, 2, 3, 4, 5)
    assert result.is_contiguous()
    torch.testing.assert_close(result, torch.from_numpy(array).permute(0, 1, 4, 2, 3))


def _program_with_user_input(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        graph_signature=SimpleNamespace(user_inputs=[name]),
        graph=SimpleNamespace(nodes=[SimpleNamespace(op="placeholder", name=name)]),
    )


def test_user_inputs_need_shape_inference_rejects_static_input(monkeypatch) -> None:
    monkeypatch.setattr(
        runner_utils,
        "get_first_fake_tensor",
        lambda node: SimpleNamespace(shape=(1, 2)),
    )

    assert not runner_utils.user_inputs_need_shape_inference(
        cast(Any, _program_with_user_input("input"))
    )


def test_user_inputs_need_shape_inference_accepts_symbolic_input(monkeypatch) -> None:
    symbolic_dim = object()
    monkeypatch.setattr(
        runner_utils,
        "get_first_fake_tensor",
        lambda node: SimpleNamespace(shape=(1, symbolic_dim)),
    )

    assert runner_utils.user_inputs_need_shape_inference(
        cast(Any, _program_with_user_input("input"))
    )


def test_user_inputs_need_shape_inference_ignores_non_user_inputs(monkeypatch) -> None:
    program = SimpleNamespace(
        graph_signature=SimpleNamespace(user_inputs=["input"]),
        graph=SimpleNamespace(
            nodes=[
                SimpleNamespace(op="placeholder", name="input"),
                SimpleNamespace(op="placeholder", name="param"),
            ]
        ),
    )

    def fake_tensor(node):
        if node.name == "input":
            return SimpleNamespace(shape=(1, 2))
        return SimpleNamespace(shape=(1, object()))

    monkeypatch.setattr(runner_utils, "get_first_fake_tensor", fake_tensor)

    assert not runner_utils.user_inputs_need_shape_inference(cast(Any, program))


def test_enable_vulkan_validation_puts_validation_first_and_deduplicates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(runner_utils.VULKAN_VALIDATION_ENV, "1")

    vulkan_sdk = tmp_path / "vulkan-sdk"
    validation_layer_dir = vulkan_sdk / "share/vulkan/explicit_layer.d"
    validation_layer_dir.mkdir(parents=True)
    (validation_layer_dir / "VkLayer_khronos_validation.json").write_text(
        "{}", encoding="utf-8"
    )

    emulation_layer_dir = tmp_path / "emulation-layers"
    emulation_layer_dir.mkdir()
    emulation_layers = ["VK_LAYER_ARM_tensor", "VK_LAYER_ARM_graph"]
    env = {
        # Include a trailing path separator to cover setup_path.sh's CI form.
        "VULKAN_SDK": f"{vulkan_sdk}{os.path.pathsep}",
        "VK_LAYER_PATH": str(emulation_layer_dir),
        "VK_ADD_LAYER_PATH": str(emulation_layer_dir),
        "VK_INSTANCE_LAYERS": os.path.pathsep.join(
            [
                emulation_layers[0],
                runner_utils.VULKAN_VALIDATION_LAYER,
                emulation_layers[1],
                runner_utils.VULKAN_VALIDATION_LAYER,
            ]
        ),
    }

    result = runner_utils._enable_vulkan_validation(env)
    layers = result["VK_INSTANCE_LAYERS"].split(os.path.pathsep)

    assert layers == [
        runner_utils.VULKAN_VALIDATION_LAYER,
        *emulation_layers,
    ]
    assert result["VK_LAYER_PATH"].split(os.path.pathsep) == [
        str(validation_layer_dir),
        str(emulation_layer_dir),
    ]
    assert result["VK_ADD_LAYER_PATH"].split(os.path.pathsep) == [
        str(validation_layer_dir),
        str(emulation_layer_dir),
    ]
    assert result["VK_KHRONOS_VALIDATION_REPORT_FLAGS"] == "error"


def test_add_known_vkml_validation_filters_uses_platform_separator() -> None:
    custom_vuid = "VUID-Test-existing-filter-00001"
    first_known_vuid = runner_utils.KNOWN_VKML_VALIDATION_VUIDS[0]
    env = {
        runner_utils.VULKAN_VALIDATION_MESSAGE_FILTER_ENV: os.path.pathsep.join(
            [custom_vuid, first_known_vuid]
        )
    }

    runner_utils._add_known_vkml_validation_filters(env)

    filters = env[runner_utils.VULKAN_VALIDATION_MESSAGE_FILTER_ENV].split(
        os.path.pathsep
    )

    assert filters == [
        custom_vuid,
        *runner_utils.KNOWN_VKML_VALIDATION_VUIDS,
    ]
    assert filters.count(first_known_vuid) == 1
