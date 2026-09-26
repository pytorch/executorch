# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Integration test for the Khronos Vulkan Validation Layer."""

from __future__ import annotations

import os
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest


_VALIDATION_LAYER = "VK_LAYER_KHRONOS_validation"
_VALIDATION_ENV = "EXECUTORCH_VGF_VULKAN_VALIDATION"
_EXPECTED_VUID = "VUID-VkApplicationInfo-sType-sType"

# We check that validation layer is enabled.
# Run the actual Vulkan call in a child process so loading Vulkan/VVL does not
# modify the pytest process.
_VULKAN_VALIDATION_PROBE = r"""
import ctypes
import ctypes.util
import os
import sys
from pathlib import Path


VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
VK_SUCCESS = 0


class VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int32),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.POINTER(VkApplicationInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.POINTER(ctypes.c_char_p)),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.POINTER(ctypes.c_char_p)),
    ]


def vulkan_library_candidates():
    candidates = []

    sdk = os.environ.get("VULKAN_SDK", "")
    if sdk:
        sdk_root = Path(sdk.split(os.pathsep)[0])

        if sys.platform == "darwin":
            candidates.extend(
                [
                    sdk_root / "lib/libvulkan.dylib",
                    sdk_root / "lib/libvulkan.1.dylib",
                ]
            )
        elif sys.platform == "win32":
            candidates.extend(
                [
                    sdk_root / "Bin/vulkan-1.dll",
                    sdk_root / "bin/vulkan-1.dll",
                ]
            )
        else:
            candidates.extend(
                [
                    sdk_root / "lib/libvulkan.so.1",
                    sdk_root / "lib/libvulkan.so",
                ]
            )

    discovered = ctypes.util.find_library("vulkan")
    if discovered:
        candidates.append(discovered)

    if sys.platform == "darwin":
        candidates.extend(["libvulkan.dylib", "libvulkan.1.dylib"])
    elif sys.platform == "win32":
        candidates.append("vulkan-1.dll")
    else:
        candidates.extend(["libvulkan.so.1", "libvulkan.so"])

    return candidates


def load_vulkan():
    errors = []

    for candidate in vulkan_library_candidates():
        try:
            return ctypes.CDLL(str(candidate))
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")

    raise RuntimeError(
        "Unable to load Vulkan loader.\n" + "\n".join(errors)
    )


vulkan = load_vulkan()

vkCreateInstance = vulkan.vkCreateInstance
vkCreateInstance.argtypes = [
    ctypes.POINTER(VkInstanceCreateInfo),
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
]
vkCreateInstance.restype = ctypes.c_int32

vkDestroyInstance = vulkan.vkDestroyInstance
vkDestroyInstance.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
]
vkDestroyInstance.restype = None


# Deliberately invalid:
#
# VkApplicationInfo requires
#
#     VK_STRUCTURE_TYPE_APPLICATION_INFO
#
# but we intentionally provide VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO.
#
# VK_LAYER_KHRONOS_validation must report:
#
#     VUID-VkApplicationInfo-sType-sType
#
application_info = VkApplicationInfo(
    sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    pNext=None,
    pApplicationName=b"executorch-vulkan-validation-probe",
    applicationVersion=0,
    pEngineName=b"executorch",
    engineVersion=0,
    apiVersion=0,
)

instance_info = VkInstanceCreateInfo(
    sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    pNext=None,
    flags=0,
    pApplicationInfo=ctypes.pointer(application_info),
    enabledLayerCount=0,
    ppEnabledLayerNames=None,
    enabledExtensionCount=0,
    ppEnabledExtensionNames=None,
)

instance = ctypes.c_void_p()

result = vkCreateInstance(
    ctypes.byref(instance_info),
    None,
    ctypes.byref(instance),
)

print(f"vkCreateInstance result={result}")

if result == VK_SUCCESS and instance.value:
    vkDestroyInstance(instance, None)
"""


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name)

    if value is None:
        return False

    return value.strip().lower() not in {
        "",
        "0",
        "false",
        "off",
        "no",
    }


def _configured_instance_layers() -> list[str]:
    value = os.environ.get("VK_INSTANCE_LAYERS", "")

    return [layer for layer in value.split(os.pathsep) if layer]


def _assert_validation_manifest_visible() -> None:
    """Check VK_LAYER_PATH when it overrides normal loader discovery."""

    layer_path = os.environ.get("VK_LAYER_PATH")

    if not layer_path:
        # The loader can use its standard installation directories.
        return

    manifest = "VkLayer_khronos_validation.json"

    candidates = [
        Path(entry) / manifest for entry in layer_path.split(os.pathsep) if entry
    ]

    assert any(path.is_file() for path in candidates), (
        "VK_LAYER_PATH is set, but it does not contain the Khronos "
        "validation-layer manifest.\n"
        f"VK_LAYER_PATH={layer_path}\n"
        "Checked:\n" + "\n".join(f"  {path}" for path in candidates)
    )


def test_vgf_vulkan_validation_layer_reports_bad_stype():
    """Verify that the real Khronos layer detects invalid Vulkan usage."""

    if not _env_flag_enabled(_VALIDATION_ENV):
        pytest.skip(f"{_VALIDATION_ENV} is not enabled for this test run")

    layers = _configured_instance_layers()

    assert _VALIDATION_LAYER in layers, (
        f"{_VALIDATION_ENV}=1, but {_VALIDATION_LAYER} is not present in "
        "VK_INSTANCE_LAYERS.\n"
        f"VK_INSTANCE_LAYERS={os.environ.get('VK_INSTANCE_LAYERS', '')}"
    )

    _assert_validation_manifest_visible()

    env = os.environ.copy()

    # Test the Khronos layer itself in isolation. We already asserted above
    # that the real VGF/VKML environment contains it alongside the emulation
    # layers.
    env["VK_INSTANCE_LAYERS"] = _VALIDATION_LAYER

    # Force deterministic diagnostic output from this negative probe.
    env["VK_KHRONOS_VALIDATION_REPORT_FLAGS"] = "error"
    env["VK_KHRONOS_VALIDATION_LOG_FILENAME"] = "stdout"
    env["VK_KHRONOS_VALIDATION_DEBUG_ACTION"] = "VK_DBG_LAYER_ACTION_LOG_MSG"

    result = subprocess.run(  # nosec B603
        [
            sys.executable,
            "-c",
            _VULKAN_VALIDATION_PROBE,
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    output = result.stdout + "\n" + result.stderr

    assert _EXPECTED_VUID in output, (
        "VK_LAYER_KHRONOS_validation was enabled but did not report the "
        "deliberately invalid VkApplicationInfo::sType.\n\n"
        f"Expected VUID:\n  {_EXPECTED_VUID}\n\n"
        f"VK_INSTANCE_LAYERS={env.get('VK_INSTANCE_LAYERS', '')}\n"
        f"VK_LAYER_PATH={env.get('VK_LAYER_PATH', '')}\n"
        f"VK_ADD_LAYER_PATH={env.get('VK_ADD_LAYER_PATH', '')}\n"
        f"VULKAN_SDK={env.get('VULKAN_SDK', '')}\n\n"
        f"Probe return code: {result.returncode}\n\n"
        f"stdout:\n{result.stdout}\n\n"
        f"stderr:\n{result.stderr}"
    )


def test_vgf_validation_wrapper_fails_on_validation_error():
    """A real VVL error must make the direct-command wrapper fail."""

    if not _env_flag_enabled(_VALIDATION_ENV):
        pytest.skip(f"{_VALIDATION_ENV} is not enabled for this test run")

    validation_wrapper = (
        Path(__file__).resolve().parents[1] / "run_with_vulkan_validation.sh"
    )

    if not validation_wrapper.is_file():
        pytest.skip(
            "Vulkan validation shell wrapper is not available in this "
            "test environment"
        )

    env = os.environ.copy()
    env[_VALIDATION_ENV] = "1"
    env["VK_INSTANCE_LAYERS"] = _VALIDATION_LAYER

    # The wrapper itself must perform the error-to-failure conversion.
    # Do not use VK_DBG_LAYER_ACTION_FAIL here.
    env["VK_KHRONOS_VALIDATION_DEBUG_ACTION"] = "VK_DBG_LAYER_ACTION_LOG_MSG"

    result = subprocess.run(  # nosec B603
        [
            str(validation_wrapper),
            sys.executable,
            "-c",
            _VULKAN_VALIDATION_PROBE,
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    output = result.stdout + "\n" + result.stderr

    assert result.returncode != 0, (
        "The Vulkan validation wrapper returned success even though the "
        "probe deliberately generated invalid Vulkan usage.\n\n"
        f"stdout:\n{result.stdout}\n\n"
        f"stderr:\n{result.stderr}"
    )

    assert _EXPECTED_VUID in output, (
        "The wrapper failed, but the expected Vulkan validation diagnostic "
        "was not observed.\n\n"
        f"Expected VUID: {_EXPECTED_VUID}\n\n"
        f"stdout:\n{result.stdout}\n\n"
        f"stderr:\n{result.stderr}"
    )


def test_vgf_validation_wrapper_preserves_error_from_earlier_vulkan_child():
    """A later clean Vulkan child must not erase an earlier validation error."""

    if not _env_flag_enabled(_VALIDATION_ENV):
        pytest.skip(f"{_VALIDATION_ENV} is not enabled for this test run")

    validation_wrapper = (
        Path(__file__).resolve().parents[1] / "run_with_vulkan_validation.sh"
    )
    if not validation_wrapper.is_file():
        pytest.skip(
            "Vulkan validation shell wrapper is not available in this "
            "test environment"
        )

    # Fix only VkApplicationInfo::sType to create a clean second probe.
    clean_probe = _VULKAN_VALIDATION_PROBE.replace(
        "sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,\n"
        "    pNext=None,\n"
        '    pApplicationName=b"executorch-vulkan-validation-probe"',
        "sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,\n"
        "    pNext=None,\n"
        '    pApplicationName=b"executorch-vulkan-validation-probe"',
        1,
    )
    assert clean_probe != _VULKAN_VALIDATION_PROBE

    # Run the invalid child first and a clean Vulkan child second. With the old
    # shared log file, the clean child could truncate the first child's VUID.
    multi_child_probe = f"""
import subprocess
import sys

invalid_probe = {_VULKAN_VALIDATION_PROBE!r}
clean_probe = {clean_probe!r}

subprocess.run([sys.executable, "-c", invalid_probe], check=True)
subprocess.run([sys.executable, "-c", clean_probe], check=True)
"""

    env = os.environ.copy()
    env[_VALIDATION_ENV] = "1"
    env["VK_INSTANCE_LAYERS"] = _VALIDATION_LAYER
    env["VK_KHRONOS_VALIDATION_DEBUG_ACTION"] = "VK_DBG_LAYER_ACTION_LOG_MSG"

    result = subprocess.run(  # nosec B603
        [str(validation_wrapper), sys.executable, "-c", multi_child_probe],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    output = result.stdout + "\n" + result.stderr

    assert result.returncode != 0, (
        "The validation wrapper returned success even though an earlier Vulkan "
        "child generated a validation error and a clean child ran afterwards.\n\n"
        f"stdout:\n{result.stdout}\n\n"
        f"stderr:\n{result.stderr}"
    )
    assert _EXPECTED_VUID in output, (
        "The validation error from the first Vulkan child was not preserved.\n\n"
        f"Expected VUID: {_EXPECTED_VUID}\n\n"
        f"stdout:\n{result.stdout}\n\n"
        f"stderr:\n{result.stderr}"
    )
