import ctypes
import getpass
import logging
import os
import platform
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Optional

import tensorrt as trt
import torch

logger = logging.getLogger(__name__)

_WHL_CPYTHON_VERSION = "cp310"
_TENSORRT_LLM_VERSION_ = "0.17.0.post1"


def sanitized_torch_version() -> Any:
    return (
        torch.__version__
        if ".nv" not in torch.__version__
        else torch.__version__.split(".nv")[0]
    )


def check_cross_compile_trt_win_lib() -> bool:
    # cross compile feature is only available on linux
    # build engine on linux and run on windows
    if sys.platform.startswith("linux"):
        import re

        import dllist

        loaded_libs = dllist.dllist()
        target_lib = ".*libnvinfer_builder_resource_win.so.*"
        return any(re.match(target_lib, lib) for lib in loaded_libs)
    return False


def is_tensorrt_version_supported(min_version: str) -> bool:
    """
    Check if the installed TensorRT version supports the specified minimum version.
    Args:
        min_version (str): Minimum required TensorRT version
    Returns:
        bool: True if TensorRT version is >= min_version, False otherwise
    Example:
        >>> if is_tensorrt_version_supported("10.8.0"):
        ...     # Use FP4 features
        ...     pass
    """
    try:
        if trt._package_name == "tensorrt_rtx":
            return True

        from packaging.version import Version

        module = sys.modules["tensorrt"]
        if module is not None and hasattr(module, "__version__"):
            return bool(Version(module.__version__) >= Version(min_version))
        # if cannot get from the modules, fall back to metadata
        from importlib import metadata

        return bool(Version(metadata.version("tensorrt")) >= Version(min_version))
    except (ImportError, ValueError):
        # If tensorrt is not installed or version cannot be determined
        return False


def is_tegra_platform() -> bool:
    if torch.cuda.get_device_capability() in [(8, 7), (7, 2)]:
        return True
    return False


def is_thor() -> bool:
    if torch.cuda.get_device_capability() in [(11, 0)]:
        return True
    return False


def is_platform_supported_for_trtllm() -> bool:
    """
    Checks if the current platform supports TensorRT-LLM plugins for the NCCL backend.

    Returns:
        bool: True if supported, False otherwise.

    Unsupported:
        - Windows platforms
        - Jetson/Orin/Xavier (aarch64 architecture + 'tegra' in platform release)
        - Thor devices
        - CUDA 13 not supported
    """
    system = platform.system().lower()
    machine = platform.machine().lower()
    release = platform.release().lower()

    if "windows" in system:
        logger.info(
            "TensorRT-LLM plugins for NCCL backend are not supported on Windows."
        )
        return False

    if machine == "aarch64" and "tegra" in release or is_thor():
        logger.info(
            "TensorRT-LLM plugins for NCCL backend are not supported on Jetson/Orin/Xavier (Tegra) or Thor devices."
        )
        return False

    try:
        cuda_version = torch.version.cuda  # e.g., "12.4" or "13.0"
        if cuda_version is None:
            logger.error(
                "This pytorch build does not support CUDA, please reinstall pytorch with CUDA support"
            )
            return False

        major, minor = map(int, cuda_version.split("."))
        if major != 12:
            logger.error(
                "CUDA 13 is not currently supported for TRT-LLM plugins. Please install pytorch with CUDA 12.x support"
            )
            return False

        return True

    except Exception as e:
        logger.warning(f"Failed to detect CUDA version: {e}")
        return False

    return True


def _cache_root() -> Path:
    username = getpass.getuser()
    return Path(tempfile.gettempdir()) / f"torch_tensorrt_{username}"


def _extracted_dir_trtllm(platform_system: str, platform_machine: str) -> Path:
    return (
        _cache_root()
        / "trtllm"
        / f"{_TENSORRT_LLM_VERSION_}_{platform_system}_{platform_machine}"
    )


def download_and_get_plugin_lib_path() -> Optional[str]:
    """
    Returns the path to the TensorRT‑LLM shared library, downloading and extracting if necessary.

    Args:
        platform (str): Platform identifier (e.g., 'linux_x86_64')

    Returns:
        Optional[str]: Path to shared library or None if operation fails.
    """
    platform_system = platform.system().lower()
    platform_machine = platform.machine().lower()
    wheel_filename = (
        f"tensorrt_llm-{_TENSORRT_LLM_VERSION_}-{_WHL_CPYTHON_VERSION}-"
        f"{_WHL_CPYTHON_VERSION}-{platform_system}_{platform_machine}.whl"
    )
    wheel_path = _cache_root() / wheel_filename
    extract_dir = _extracted_dir_trtllm(platform_system, platform_machine)
    # else will never be met though
    lib_filename = (
        "libnvinfer_plugin_tensorrt_llm.so"
        if "linux" in platform_system
        else "libnvinfer_plugin_tensorrt_llm.dll"
    )
    # eg: /tmp/torch_tensorrt_<username>/trtllm/0.17.0.post1_linux_x86_64/tensorrt_llm/libs/libnvinfer_plugin_tensorrt_llm.so
    plugin_lib_path = extract_dir / "tensorrt_llm" / "libs" / lib_filename

    if plugin_lib_path.exists():
        return str(plugin_lib_path)

    wheel_path.parent.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    if not wheel_path.exists():
        base_url = "https://pypi.nvidia.com/tensorrt-llm/"
        download_url = base_url + wheel_filename
        try:
            logger.debug(f"Downloading {download_url} ...")
            urllib.request.urlretrieve(download_url, wheel_path)
            logger.debug("Download succeeded and TRT-LLM wheel is now present")
        except urllib.error.HTTPError as e:
            logger.error(
                f"HTTP error {e.code} when trying to download {download_url}: {e.reason}"
            )
        except urllib.error.URLError as e:
            logger.error(
                f"URL error when trying to download {download_url}: {e.reason}"
            )
        except OSError as e:
            logger.error(f"Local file write error: {e}")

    try:
        import zipfile
    except ImportError as e:
        raise ImportError(
            "zipfile module is required but not found. Please install zipfile"
        )
    try:
        with zipfile.ZipFile(wheel_path) as zip_ref:
            zip_ref.extractall(extract_dir)
            logger.debug(f"Extracted wheel to {extract_dir}")
    except FileNotFoundError as e:
        # This should capture the errors in the download failure above
        logger.error(f"Wheel file not found at {wheel_path}: {e}")
        raise RuntimeError(
            f"Failed to find downloaded wheel file at {wheel_path}"
        ) from e
    except zipfile.BadZipFile as e:
        logger.error(f"Invalid or corrupted wheel file: {e}")
        raise RuntimeError(
            "Downloaded wheel file is corrupted or not a valid zip archive"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error while extracting wheel: {e}")
        raise RuntimeError(
            "Unexpected error during extraction of TensorRT-LLM wheel"
        ) from e

    try:
        wheel_path.unlink(missing_ok=True)
        logger.debug(f"Deleted wheel file: {wheel_path}")
    except Exception as e:
        logger.warning(f"Could not delete wheel file {wheel_path}: {e}")
    if not plugin_lib_path.exists():
        logger.error(
            f"Plugin library not found at expected location: {plugin_lib_path}"
        )
        return None

    return str(plugin_lib_path)


def load_and_initialize_trtllm_plugin(plugin_lib_path: str) -> bool:
    """
    Loads and initializes the TensorRT-LLM plugin from the given shared library path.

    Args:
        plugin_lib_path (str): Path to the shared TensorRT-LLM plugin library.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        handle = ctypes.CDLL(plugin_lib_path)
        logger.info(f"Successfully loaded plugin library: {plugin_lib_path}")
    except OSError as e_os_error:
        if "libmpi" in str(e_os_error):
            logger.warning(
                f"Failed to load libnvinfer_plugin_tensorrt_llm.so from {plugin_lib_path}, got error {e_os_error} (hint: libmpi.so is a necessary dependency; ensure that OpenMPI or MPICH is installed on your system)",
                exc_info=e_os_error,
            )
        else:
            logger.warning(
                f"Failed to load libnvinfer_plugin_tensorrt_llm.so from {plugin_lib_path}. "
                f"Ensure the path is correct and the library is compatible.",
                exc_info=e_os_error,
            )
        return False

    try:
        handle.initTrtLlmPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        handle.initTrtLlmPlugins.restype = ctypes.c_bool
    except AttributeError as e_plugin_unavailable:
        logger.warning(
            "Unable to initialize the TensorRT-LLM plugin library",
            exc_info=e_plugin_unavailable,
        )
        return False

    try:
        if handle.initTrtLlmPlugins(None, b"tensorrt_llm"):
            logger.info("TensorRT-LLM plugin successfully initialized")
            return True
        else:
            logger.warning("TensorRT-LLM plugin library failed in initialization")
            return False
    except Exception as e_initialization_error:
        logger.warning(
            "Exception occurred during TensorRT-LLM plugin library initialization",
            exc_info=e_initialization_error,
        )
        return False
    return False


def load_tensorrt_llm_for_nccl() -> bool:
    """
    Attempts to load the TensorRT-LLM plugin and initialize it.
    Either the env variable TRTLLM_PLUGINS_PATH can specify the path
    Or the user can specify USE_TRTLLM_PLUGINS as either of (1, true, yes, on) to download the TRT-LLM distribution and load it

    Returns:
        bool: True if the plugin was successfully loaded and initialized, False otherwise.
    """
    if not is_platform_supported_for_trtllm():
        return False
    plugin_lib_path = os.environ.get("TRTLLM_PLUGINS_PATH")

    if plugin_lib_path:
        return load_and_initialize_trtllm_plugin(plugin_lib_path)
    else:
        # this option can be used by user if TRTLLM_PLUGINS_PATH is not set by user
        use_trtllm_plugin = os.environ.get("USE_TRTLLM_PLUGINS", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not use_trtllm_plugin:
            logger.warning(
                "Neither TRTLLM_PLUGIN_PATH is set nor is it directed to download the shared library. Please set either of the two to use TRT-LLM libraries in torchTRT"
            )
            return False

        plugin_lib_path = download_and_get_plugin_lib_path()
        return load_and_initialize_trtllm_plugin(plugin_lib_path)  # type: ignore[arg-type]
    return False
