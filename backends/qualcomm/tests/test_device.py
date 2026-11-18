# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import call, patch

import pytest

from executorch.backends.qualcomm.export_utils import ADB, Device, WindowsBridge
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.utils.utils import get_soc_to_htp_arch_map


def test_adb_commands_match_legacy_behavior():
    completed = subprocess.CompletedProcess([], 0)
    with patch(
        "executorch.backends.qualcomm.export_utils.subprocess.run",
        return_value=completed,
    ) as run:
        bridge = ADB("device-id", "host-id")
        bridge.mkdir("/workspace")
        bridge.rmdir("/workspace")
        bridge.push_file("model.pte", "/workspace")
        bridge.pull_file("/workspace/outputs", "outputs", recursive=True)
        bridge.run("echo test")
        bridge.run_executor("/workspace", "/build/runner", "--model model.pte")

    prefix = ["adb", "-H", "host-id", "-s", "device-id"]
    runner_command = (
        "cd /workspace && chmod +x runner && export LD_LIBRARY_PATH=. && "
        "export ADSP_LIBRARY_PATH=. && echo 0x0C > runner.farf && "
        "./runner --model model.pte"
    )
    assert run.call_args_list == [
        call(prefix + ["shell", "mkdir -p /workspace"], stdout=sys.__stdout__),
        call(prefix + ["shell", "rm -rf /workspace"], stdout=sys.__stdout__),
        call(prefix + ["push", "model.pte", "/workspace"], stdout=sys.__stdout__),
        call(
            prefix + ["pull", "-a", "/workspace/outputs", "outputs"],
            stdout=sys.__stdout__,
        ),
        call(prefix + ["shell", "echo test"], stdout=sys.__stdout__),
        call(prefix + ["shell", runner_command], stdout=sys.__stdout__),
    ]


def test_adb_command_without_host_matches_legacy_behavior():
    completed = subprocess.CompletedProcess([], 0)
    with patch(
        "executorch.backends.qualcomm.export_utils.subprocess.run",
        return_value=completed,
    ) as run:
        ADB("device-id").run("echo test")

    run.assert_called_once_with(
        ["adb", "-s", "device-id", "shell", "echo test"], stdout=sys.__stdout__
    )


def test_output_callback_receives_stdout_after_success():
    completed = subprocess.CompletedProcess([], 0, stdout="command output")
    with patch(
        "executorch.backends.qualcomm.export_utils.subprocess.run",
        return_value=completed,
    ):
        callback_output = []
        ADB("device-id").run("echo test", callback_output.append)

    assert callback_output == ["command output"]


def test_output_callback_is_not_called_on_failure():
    completed = subprocess.CompletedProcess([], 1, stdout="command output")
    with patch(
        "executorch.backends.qualcomm.export_utils.subprocess.run",
        return_value=completed,
    ):
        callback_output = []
        with pytest.raises(RuntimeError, match="device command failed"):
            ADB("device-id").run("false", callback_output.append)

    assert callback_output == []


def test_windows_bridge_commands_pass_paths_as_parameters():
    completed = subprocess.CompletedProcess([], 0)
    workspace = 'C:\\workspace-$value-"quoted"'
    source = 'C:\\source-$value-"quoted"'
    destination = 'C:\\destination-$value-"quoted"'
    with patch(
        "executorch.backends.qualcomm.export_utils.subprocess.run",
        return_value=completed,
    ) as run:
        bridge = WindowsBridge()
        bridge.mkdir(workspace)
        bridge.rmdir(workspace)
        bridge.push_file(source, workspace)
        bridge.pull_file(source, destination, recursive=True)
        bridge.run("Write-Output test")
        bridge.run_executor(workspace, "C:/build/runner.exe", "--model model.pte")

    assert run.call_args_list == [
        call(
            [
                "powershell",
                "-Command",
                "& { param([string]$Path) New-Item -ItemType Directory -Force "
                "-LiteralPath $Path | Out-Null }",
                "-Path",
                workspace,
            ],
            stdout=sys.__stdout__,
        ),
        call(
            [
                "powershell",
                "-Command",
                "& { param([string]$Path) if (Test-Path -LiteralPath $Path) { "
                "Remove-Item -LiteralPath $Path -Recurse -Force } }",
                "-Path",
                workspace,
            ],
            stdout=sys.__stdout__,
        ),
        call(
            [
                "powershell",
                "-Command",
                "& { param([string]$Source, [string]$Destination) Copy-Item "
                "-LiteralPath $Source -Destination $Destination -Force }",
                "-Source",
                source,
                "-Destination",
                workspace,
            ],
            stdout=sys.__stdout__,
        ),
        call(
            [
                "powershell",
                "-Command",
                "& { param([string]$Source, [string]$Destination) Copy-Item "
                "-LiteralPath $Source -Destination $Destination -Recurse -Force }",
                "-Source",
                source,
                "-Destination",
                destination,
            ],
            stdout=sys.__stdout__,
        ),
        call(
            ["powershell", "-Command", "Write-Output test"],
            stdout=sys.__stdout__,
        ),
        call(
            [
                "powershell",
                "-Command",
                "& { param([string]$Workspace, [string]$Runner) "
                "Set-Location -LiteralPath "
                '$Workspace; $env:PATH = ".;" + $env:PATH; '
                '$env:ADSP_LIBRARY_PATH="."; '
                '& (Join-Path "." $Runner) --model model.pte }',
                "-Workspace",
                workspace,
                "-Runner",
                "runner.exe",
            ],
            stdout=sys.__stdout__,
        ),
    ]


def _qnn_config(target, backend):
    return SimpleNamespace(
        target=target,
        backend=backend,
        build_folder="/build",
        direct_build_folder=None,
        pre_gen_pte=None,
        device="device-id",
        host=None,
        dump_intermediate_outputs=False,
        soc_model="SM8850",
        shared_buffer=False,
        skip_push=False,
    )


@pytest.mark.parametrize(
    "target,bridge_type,runner,system_library,backend_library",
    [
        (
            "aarch64-android",
            ADB,
            "examples/qualcomm/executor_runner/qnn_executor_runner",
            "/qnn/lib/aarch64-android/libQnnSystem.so",
            "/build/backends/qualcomm/libqnn_executorch_backend.so",
        ),
        (
            "aarch64-oe-linux-gcc9.3",
            ADB,
            "examples/qualcomm/executor_runner/qnn_executor_runner",
            "/qnn/lib/aarch64-oe-linux-gcc9.3/libQnnSystem.so",
            "/build/backends/qualcomm/libqnn_executorch_backend.so",
        ),
        (
            "aarch64-oe-linux-gcc11.2",
            ADB,
            "examples/qualcomm/executor_runner/qnn_executor_runner",
            "/qnn/lib/aarch64-oe-linux-gcc11.2/libQnnSystem.so",
            "/build/backends/qualcomm/libqnn_executorch_backend.so",
        ),
        (
            "aarch64-windows-msvc",
            WindowsBridge,
            "examples/qualcomm/executor_runner/Release/qnn_executor_runner.exe",
            "/qnn/lib/aarch64-windows-msvc/QnnSystem.dll",
            "/build/backends/qualcomm/Release/qnn_executorch_backend.dll",
        ),
        (
            "x86_64-windows-msvc",
            WindowsBridge,
            "examples/qualcomm/executor_runner/Release/qnn_executor_runner.exe",
            "/qnn/lib/x86_64-windows-msvc/QnnSystem.dll",
            "/build/backends/qualcomm/Release/qnn_executorch_backend.dll",
        ),
    ],
)
def test_device_selects_bridge_runner_and_common_libraries(
    target, bridge_type, runner, system_library, backend_library
):
    # This test runs on a Linux host, so host artifact paths use POSIX
    # separators. Device workspace paths use the target's path convention.
    workspace = "C:\\workspace" if target.endswith("windows-msvc") else "/workspace"
    with patch.dict("os.environ", {"QNN_SDK_ROOT": "/qnn"}):
        device = Device(
            _qnn_config(target, QnnExecuTorchBackendType.kHtpBackend),
            "model.pte",
            workspace,
        )

    assert isinstance(device.bridge, bridge_type)
    assert device.runner == runner
    separator = "\\" if target.endswith("windows-msvc") else "/"
    assert device.etdump_path == f"{workspace}{separator}etdump.etdp"
    assert device.debug_output_path == f"{workspace}{separator}debug_output.bin"
    assert device.output_folder == f"{workspace}{separator}outputs"
    htp_arch = get_soc_to_htp_arch_map()["SM8850"]
    if target.endswith("windows-msvc"):
        expected_htp_libraries = [
            f"/qnn/lib/{target}/QnnHtp.dll",
            f"/qnn/lib/hexagon-v{htp_arch}/unsigned/libQnnHtpV{htp_arch}Skel.so",
            f"/qnn/lib/hexagon-v{htp_arch}/unsigned/libqnnhtpv{htp_arch}.cat",
            f"/qnn/lib/{target}/QnnHtpV{htp_arch}Stub.dll",
            f"/qnn/lib/{target}/QnnHtpPrepare.dll",
        ]
        common_libraries = [system_library, backend_library]
        expected_libraries = {
            QnnExecuTorchBackendType.kHtpBackend: expected_htp_libraries
            + common_libraries
        }
    else:
        expected_htp_libraries = [
            f"/qnn/lib/{target}/libQnnHtp.so",
            f"/qnn/lib/hexagon-v{htp_arch}/unsigned/libQnnHtpV{htp_arch}Skel.so",
            f"/qnn/lib/{target}/libQnnHtpV{htp_arch}Stub.so",
            f"/qnn/lib/{target}/libQnnHtpPrepare.so",
        ]
        expected_gpu_libraries = [f"/qnn/lib/{target}/libQnnGpu.so"]
        expected_lpai_libraries = [
            f"/qnn/lib/{target}/libQnnLpai.so",
            f"/qnn/lib/{target}/libQnnLpaiStub.so",
        ]

        common_libraries = [system_library, backend_library]
        expected_libraries = {
            QnnExecuTorchBackendType.kHtpBackend: expected_htp_libraries
            + common_libraries,
            QnnExecuTorchBackendType.kGpuBackend: expected_gpu_libraries
            + common_libraries,
            QnnExecuTorchBackendType.kLpaiBackend: expected_lpai_libraries
            + common_libraries,
        }

    assert device.backend_library_paths == expected_libraries


def test_device_reports_unsupported_target_backend_pair():
    with patch.dict("os.environ", {"QNN_SDK_ROOT": "/qnn"}):
        device = Device(
            _qnn_config("aarch64-windows-msvc", QnnExecuTorchBackendType.kGpuBackend),
            "model.pte",
            "C:\\workspace",
        )

    with pytest.raises(
        ValueError, match="kGpuBackend is not supported for target aarch64-windows-msvc"
    ):
        device._library_paths_for(QnnExecuTorchBackendType.kGpuBackend)
