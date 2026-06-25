# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import stat
from pathlib import Path

import executorch.backends.arm.vgf.check_env as check_env
import executorch.backends.arm.vgf.model_converter as model_converter

import pytest
from executorch.backends.arm.vgf import backend as vgf_backend
from executorch.backends.arm.vgf.compile_spec import VgfCompileSpec


def _make_executable(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _pass(name: str = "ok") -> check_env.VgfEnvironmentCheck:
    return check_env.VgfEnvironmentCheck(name, check_env.STATUS_OK, "ok")


def _fail(name: str = "bad") -> check_env.VgfEnvironmentCheck:
    return check_env.VgfEnvironmentCheck(name, check_env.STATUS_FAIL, "bad", "fix it")


def test_aot_environment_uses_only_aot_checks(monkeypatch):
    monkeypatch.setattr(check_env, "_check_tosa_serializer", lambda: _pass("tosa"))
    monkeypatch.setattr(check_env, "_check_model_converter", lambda: _pass("converter"))
    monkeypatch.setattr(
        check_env, "_check_model_converter_lib_dir", lambda: _pass("lib-dir")
    )

    report = check_env.check_vgf_aot_environment()

    assert report.mode == "aot"
    assert report.ok
    assert [check.name for check in report.checks] == [
        "tosa",
        "converter",
        "lib-dir",
    ]


def test_runtime_environment_uses_runtime_check(monkeypatch):
    monkeypatch.setattr(
        check_env, "_check_runtime_vgf_backend", lambda: _pass("runtime")
    )

    report = check_env.check_vgf_runtime_environment()

    assert report.mode == "runtime"
    assert report.ok
    assert [check.name for check in report.checks] == ["runtime"]


def test_host_emulator_environment_checks_runtime_vulkan_and_vkml(monkeypatch):
    monkeypatch.setattr(
        check_env, "_check_runtime_vgf_backend", lambda: _pass("runtime")
    )
    monkeypatch.setattr(check_env, "_check_vulkan_sdk", lambda: _pass("vulkan"))
    monkeypatch.setattr(check_env, "_check_emulation_layer", lambda: _pass("emulation"))

    report = check_env.check_vgf_host_emulator_environment()

    assert report.mode == "host-emulator"
    assert report.ok
    assert [check.name for check in report.checks] == [
        "runtime",
        "vulkan",
        "emulation",
    ]


def test_source_build_environment_checks_vgf_lib_and_cmake(monkeypatch):
    captured = {}

    def fake_cmake(build_dir, require_runtime_build):
        captured["build_dir"] = build_dir
        captured["require_runtime_build"] = require_runtime_build
        return _pass("cmake")

    monkeypatch.setattr(check_env, "_check_vgf_library_path", lambda: _pass("libvgf"))
    monkeypatch.setattr(check_env, "_check_cmake_build_flags", fake_cmake)

    report = check_env.check_vgf_source_build_environment(build_dir="cmake-out-vkml")

    assert report.mode == "source-build"
    assert report.ok
    assert [check.name for check in report.checks] == ["libvgf", "cmake"]
    assert captured == {
        "build_dir": "cmake-out-vkml",
        "require_runtime_build": True,
    }


def test_is_vgf_aot_available(monkeypatch):
    monkeypatch.setattr(
        check_env,
        "check_vgf_aot_environment",
        lambda: check_env.VgfEnvironmentReport([_pass()], mode="aot"),
    )

    assert check_env.is_vgf_aot_available()


def test_is_vgf_runtime_available(monkeypatch):
    monkeypatch.setattr(
        check_env,
        "check_vgf_runtime_environment",
        lambda: check_env.VgfEnvironmentReport([_pass()], mode="runtime"),
    )

    assert check_env.is_vgf_runtime_available()


def test_model_converter_check_fails_when_missing(monkeypatch):
    monkeypatch.setattr(model_converter, "find_model_converter_binary", lambda: None)

    result = check_env._check_model_converter()

    assert result.status == check_env.STATUS_FAIL
    assert "model-converter" in result.detail
    assert result.action is not None


def test_model_converter_check_reports_version(monkeypatch, tmp_path):
    converter = _make_executable(
        tmp_path / "model-converter",
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if '--version' in sys.argv:\n"
        "    print('model-converter 0.9.0')\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n",
    )
    monkeypatch.setattr(
        model_converter, "find_model_converter_binary", lambda: str(converter)
    )

    result = check_env._check_model_converter()

    assert result.status == check_env.STATUS_OK
    assert str(converter) in result.detail
    assert "0.9.0" in result.detail


def test_model_converter_lib_dir_fails_when_invalid(monkeypatch, tmp_path):
    missing = tmp_path / "missing"
    monkeypatch.setenv("MODEL_CONVERTER_LIB_DIR", str(missing))

    result = check_env._check_model_converter_lib_dir()

    assert result.status == check_env.STATUS_FAIL
    assert str(missing) in result.detail


def test_find_existing_lib_finds_libvgf(tmp_path):
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    libvgf = lib_dir / "libvgf.a"
    libvgf.write_bytes(b"fake")

    found = check_env._find_existing_lib([lib_dir], ("libvgf.a",))

    assert found == [libvgf]


def test_runtime_backend_check_passes_when_vgf_registered(monkeypatch):
    class BackendRegistry:
        registered_backend_names = [vgf_backend.VGF_BACKEND_NAME]

        def is_available(self, backend_name):
            return backend_name == vgf_backend.VGF_BACKEND_NAME

    class Runtime:
        backend_registry = BackendRegistry()

    monkeypatch.setattr(vgf_backend, "_load_runtime", lambda: Runtime())

    result = check_env._check_runtime_vgf_backend()

    assert result.status == check_env.STATUS_OK
    assert vgf_backend.VGF_BACKEND_NAME in result.detail


def test_runtime_backend_check_fails_when_vgf_not_registered(monkeypatch):
    class BackendRegistry:
        registered_backend_names = ["XnnpackBackend"]

        def is_available(self, backend_name):
            return False

    class Runtime:
        backend_registry = BackendRegistry()

    monkeypatch.setattr(vgf_backend, "_load_runtime", lambda: Runtime())

    result = check_env._check_runtime_vgf_backend()

    assert result.status == check_env.STATUS_FAIL
    assert vgf_backend.VGF_BACKEND_NAME in result.detail
    assert "XnnpackBackend" in result.detail


def test_cmake_build_flags_pass(tmp_path):
    (tmp_path / "CMakeCache.txt").write_text(
        "EXECUTORCH_BUILD_VGF:BOOL=ON\n" "EXECUTORCH_BUILD_VULKAN:BOOL=TRUE\n",
        encoding="utf-8",
    )

    result = check_env._check_cmake_build_flags(
        build_dir=tmp_path,
        require_runtime_build=True,
    )

    assert result.status == check_env.STATUS_OK
    assert "EXECUTORCH_BUILD_VGF=ON" in result.detail
    assert "EXECUTORCH_BUILD_VULKAN=TRUE" in result.detail


def test_cmake_build_flags_pass_when_vulkan_disabled(tmp_path):
    (tmp_path / "CMakeCache.txt").write_text(
        "EXECUTORCH_BUILD_VGF:BOOL=ON\n" "EXECUTORCH_BUILD_VULKAN:BOOL=OFF\n",
        encoding="utf-8",
    )

    result = check_env._check_cmake_build_flags(
        build_dir=tmp_path,
        require_runtime_build=True,
    )

    assert result.status == check_env.STATUS_OK
    assert "EXECUTORCH_BUILD_VGF=ON" in result.detail
    assert "EXECUTORCH_BUILD_VULKAN=OFF" in result.detail


def test_cmake_build_flags_fail_when_vgf_disabled(tmp_path):
    (tmp_path / "CMakeCache.txt").write_text(
        "EXECUTORCH_BUILD_VGF:BOOL=OFF\n" "EXECUTORCH_BUILD_VULKAN:BOOL=ON\n",
        encoding="utf-8",
    )

    result = check_env._check_cmake_build_flags(
        build_dir=tmp_path,
        require_runtime_build=True,
    )

    assert result.status == check_env.STATUS_FAIL
    assert "EXECUTORCH_BUILD_VGF" in result.detail
    assert result.action is not None
    assert "-DEXECUTORCH_BUILD_VGF=ON" in result.action


def test_cmake_build_flags_warn_when_runtime_build_not_required(tmp_path):
    result = check_env._check_cmake_build_flags(
        build_dir=None,
        require_runtime_build=False,
        search_roots=[tmp_path],
    )

    assert result.status == check_env.STATUS_WARN


def test_report_raise_for_errors():
    report = check_env.VgfEnvironmentReport([_fail()])

    with pytest.raises(RuntimeError, match="bad"):
        report.raise_for_errors()


def test_compile_spec_validate_environment_delegates_to_aot(monkeypatch):
    class DummyReport:
        def __init__(self):
            self.raise_called = False

        def raise_for_errors(self):
            self.raise_called = True

    report = DummyReport()
    monkeypatch.setattr(check_env, "check_vgf_aot_environment", lambda: report)

    result = VgfCompileSpec().validate_environment()

    assert result is report
    assert report.raise_called


def test_compile_spec_validate_environment_can_run_source_build(monkeypatch):
    class DummyReport:
        def __init__(self):
            self.raise_called = False

        def raise_for_errors(self):
            self.raise_called = True

    captured = {}
    report = DummyReport()

    def fake_source_build(build_dir):
        captured["build_dir"] = build_dir
        return report

    monkeypatch.setattr(
        check_env, "check_vgf_source_build_environment", fake_source_build
    )

    result = VgfCompileSpec().validate_environment(
        build_dir="cmake-out-vkml",
        require_runtime_build=True,
    )

    assert result is report
    assert report.raise_called
    assert captured == {"build_dir": "cmake-out-vkml"}


def test_main_defaults_to_aot(monkeypatch, capsys):
    monkeypatch.setattr(
        check_env,
        "check_vgf_aot_environment",
        lambda: check_env.VgfEnvironmentReport([_pass("aot")], mode="aot"),
    )

    assert check_env.main([]) == 0
    assert "aot" in capsys.readouterr().out


def test_main_runtime_mode(monkeypatch, capsys):
    monkeypatch.setattr(
        check_env,
        "check_vgf_runtime_environment",
        lambda: check_env.VgfEnvironmentReport([_pass("runtime")], mode="runtime"),
    )

    assert check_env.main(["--runtime"]) == 0
    assert "runtime" in capsys.readouterr().out


def test_main_source_build_mode(monkeypatch, capsys):
    monkeypatch.setattr(
        check_env,
        "check_vgf_source_build_environment",
        lambda build_dir: check_env.VgfEnvironmentReport(
            [_pass(str(build_dir))], mode="source-build"
        ),
    )

    assert check_env.main(["--source-build", "--build-dir", "cmake-out-vkml"]) == 0
    assert "source-build" in capsys.readouterr().out


def test_main_rejects_build_dir_without_source_build():
    with pytest.raises(SystemExit):
        check_env.main(["--build-dir", "cmake-out-vkml"])


def test_check_env_model_converter_probe_delegates_to_model_converter_module(
    monkeypatch,
):
    monkeypatch.setattr(
        model_converter,
        "check_model_converter_environment",
        lambda: model_converter.ModelConverterEnvironmentCheck(
            "converter", model_converter.STATUS_OK, "from-owner"
        ),
    )

    result = check_env._check_model_converter()

    assert result.status == check_env.STATUS_OK
    assert result.detail == "from-owner"


def test_check_env_model_converter_lib_dir_probe_delegates_to_model_converter_module(
    monkeypatch,
):
    monkeypatch.setattr(
        model_converter,
        "check_model_converter_lib_dir_environment",
        lambda: model_converter.ModelConverterEnvironmentCheck(
            "lib-dir", model_converter.STATUS_OK, "from-owner"
        ),
    )

    result = check_env._check_model_converter_lib_dir()

    assert result.status == check_env.STATUS_OK
    assert result.detail == "from-owner"


def test_check_env_runtime_probe_delegates_to_backend_module(monkeypatch):
    monkeypatch.setattr(
        vgf_backend,
        "check_vgf_runtime_backend_environment",
        lambda: vgf_backend.VgfRuntimeEnvironmentCheck(
            "runtime", vgf_backend.STATUS_OK, "from-owner"
        ),
    )

    result = check_env._check_runtime_vgf_backend()

    assert result.status == check_env.STATUS_OK
    assert result.detail == "from-owner"


def test_model_converter_preflight_and_vgf_compile_share_executable_resolution(
    monkeypatch,
    tmp_path,
):
    converter = _make_executable(
        tmp_path / "model-converter",
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import sys\n"
        "\n"
        "if '--version' in sys.argv:\n"
        "    print('model-converter integration-test')\n"
        "    raise SystemExit(0)\n"
        "\n"
        "out_index = sys.argv.index('-o') + 1\n"
        "Path(sys.argv[out_index]).write_bytes(b'compiled-vgf')\n"
        "raise SystemExit(0)\n",
    )

    monkeypatch.setenv("MODEL_CONVERTER_PATH", str(converter))

    preflight = check_env._check_model_converter()
    compiled = vgf_backend.vgf_compile(
        tosa_flatbuffer=b"fake-tosa-flatbuffer",
        compile_flags=[],
    )

    assert preflight.status == check_env.STATUS_OK
    assert str(converter) in preflight.detail
    assert compiled == b"compiled-vgf"
