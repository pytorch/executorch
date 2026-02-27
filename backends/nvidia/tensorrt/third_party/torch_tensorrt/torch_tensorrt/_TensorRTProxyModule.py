import ctypes
import importlib
import importlib.util
import logging
import os
import platform
import sys
import tempfile
from types import ModuleType
from typing import Any, Dict, List

_LOGGER = logging.getLogger(__name__)
package_imported = False
package_name = ""


def _parse_semver(version: str) -> Dict[str, str]:
    split = version.split(".")
    if len(split) < 3:
        split.append("")

    return {"major": split[0], "minor": split[1], "patch": split[2]}


def _find_lib(name: str, paths: List[str]) -> str:
    for path in paths:
        libpath = os.path.join(path, name)
        if os.path.isfile(libpath):
            return libpath

    raise FileNotFoundError(f"Could not find {name}\n  Search paths: {paths}")


def enable_capture_tensorrt_api_recording() -> None:

    os_env_flag = os.environ.get("TORCHTRT_ENABLE_TENSORRT_API_CAPTURE", None)
    if os_env_flag is None or (os_env_flag != "1" and os_env_flag.lower() != "true"):
        _LOGGER.debug("Capturing TensorRT API calls is not enabled")
        return
    if not sys.platform.startswith("linux"):
        _LOGGER.warning(
            f"Capturing TensorRT API calls is only supported on Linux, therefore ignoring the capture_tensorrt_api_recording setting for {sys.platform}"
        )
        os.environ.pop("TORCHTRT_ENABLE_TENSORRT_API_CAPTURE")
        return

    linux_lib_path = []
    if "LD_LIBRARY_PATH" in os.environ:
        linux_lib_path.extend(os.environ["LD_LIBRARY_PATH"].split(os.path.pathsep))

    if platform.uname().processor == "x86_64":
        linux_lib_path.append("/usr/lib/x86_64-linux-gnu")
    elif platform.uname().processor == "aarch64":
        linux_lib_path.append("/usr/lib/aarch64-linux-gnu")

    for path in linux_lib_path:
        if os.path.isfile(os.path.join(path, "libtensorrt_shim.so")):
            try:
                ctypes.CDLL(
                    os.path.join(path, "libtensorrt_shim.so"), mode=ctypes.RTLD_GLOBAL
                )
                tensorrt_lib_path = path
                break
            except Exception as e:
                continue

    if tensorrt_lib_path is None:
        _LOGGER.warning(
            "Capturing TensorRT API calls is enabled, but libtensorrt_shim.so is not found, make sure TensorRT lib is in the LD_LIBRARY_PATH, therefore ignoring the capture_tensorrt_api_recording setting"
        )
        os.environ.pop("TORCHTRT_ENABLE_TENSORRT_API_CAPTURE")
    else:
        os.environ["TRT_SHIM_NVINFER_LIB_NAME"] = os.path.join(
            tensorrt_lib_path, "libnvinfer.so"
        )
        import pwd

        current_user = pwd.getpwuid(os.getuid())[0]
        shim_temp_dir = os.path.join(
            tempfile.gettempdir(), f"torch_tensorrt_{current_user}/shim"
        )
        os.makedirs(shim_temp_dir, exist_ok=True)
        json_file_name = os.path.join(shim_temp_dir, "shim.json")
        os.environ["TRT_SHIM_OUTPUT_JSON_FILE"] = json_file_name
        bin_file_name = os.path.join(shim_temp_dir, "shim.bin")
        # if exists, delete the file, so that we can capture the new one
        if os.path.exists(json_file_name):
            os.remove(json_file_name)
        if os.path.exists(bin_file_name):
            os.remove(bin_file_name)
        _LOGGER.info(
            f"Capturing TensorRT API calls feature is enabled and the captured output is in the {shim_temp_dir} directory"
        )


# TensorRTProxyModule is a proxy module that allows us to register the tensorrt or tensorrt-rtx package
# since tensorrt-rtx is the drop-in replacement for tensorrt, we can use the same interface to use tensorrt-rtx
class TensorRTProxyModule(ModuleType):
    def __init__(self, target_module: ModuleType) -> None:
        spec = importlib.util.spec_from_loader("tensorrt", loader=None)
        self.__spec__ = spec
        self.__package__ = target_module.__package__
        self.__path__ = target_module.__path__
        self.__file__ = target_module.__file__
        self.__loader__ = target_module.__loader__
        self.__version__ = target_module.__version__
        self._target_module = target_module
        self._nested_module = None
        self._package_name: str = ""

        # For RTX: tensorrt.tensorrt -> tensorrt_rtx.tensorrt_rtx
        # For standard: tensorrt.tensorrt -> tensorrt.tensorrt (no change)
        if hasattr(target_module, "tensorrt_rtx"):
            self._nested_module = target_module.tensorrt_rtx
        elif hasattr(target_module, "tensorrt"):
            self._nested_module = target_module.tensorrt

        # Set up the nested module structure
        if self._nested_module:
            self.tensorrt = self._nested_module

    # __getattr__ is used to get the attribute from the target module
    def __getattr__(self, name: str) -> Any:
        # First try to get from the target module
        try:
            return getattr(self._target_module, name)
        except AttributeError:
            print(f"AttributeError: {name}")
            # For nested modules like tensorrt.tensorrt
            if name == "tensorrt" and self._nested_module:
                return self._nested_module
            raise

    def __dir__(self) -> list[str]:
        return dir(self._target_module)


def alias_tensorrt() -> None:
    global package_imported
    global package_name
    # tensorrt package has been imported, no need to alias again
    if package_imported:
        return

    # in order not to break or change the existing behavior, we only build and run with tensorrt by default, tensorrt-rtx is for experiment only
    # if we want to test with tensorrt-rtx, we have to build the wheel with --use-rtx and test with USE_TRT_RTX=true
    # eg: USE_TRT_RTX=true python test.py
    # in future, we can do dynamic linking either to tensorrt or tensorrt-rtx based on the gpu type
    use_rtx = False
    if (use_rtx_env_var := os.environ.get("USE_TRT_RTX")) is not None:
        if use_rtx_env_var.lower() == "true":
            use_rtx = True
    package_name = "tensorrt_rtx" if use_rtx else "tensorrt"

    if not use_rtx:
        # enable capture tensorrt api recording has to be done before importing the tensorrt library
        enable_capture_tensorrt_api_recording()

    # Import the appropriate package
    try:
        target_module = importlib.import_module(package_name)
        proxy = TensorRTProxyModule(target_module)
        proxy._package_name = package_name
        sys.modules["tensorrt"] = proxy
        package_imported = True
    except ImportError as e:
        # Fallback to standard tensorrt if RTX version not available
        print(f"import error when try to import {package_name=} got error {e}")
        print(
            f"make sure tensorrt lib is in the LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}"
        )
        if use_rtx:
            from torch_tensorrt._version import __tensorrt_rtx_version__

            tensorrt_version = _parse_semver(__tensorrt_rtx_version__)
            tensorrt_major = tensorrt_version["major"]
            tensorrt_minor = tensorrt_version["minor"]
            tensorrt_lib = {
                "win": [
                    f"tensorrt_rtx_{tensorrt_major}_{tensorrt_minor}.dll",
                ],
                "linux": [
                    f"libtensorrt_rtx.so.{tensorrt_major}",
                ],
            }
        else:
            from torch_tensorrt._version import __tensorrt_version__

            tensorrt_version = _parse_semver(__tensorrt_version__)
            tensorrt_major = tensorrt_version["major"]
            tensorrt_minor = tensorrt_version["minor"]
            tensorrt_lib = {
                "win": [
                    f"nvinfer_{tensorrt_major}.dll",
                    f"nvinfer_plugin_{tensorrt_major}.dll",
                ],
                "linux": [
                    f"libnvinfer.so.{tensorrt_major}",
                    f"libnvinfer_plugin.so.{tensorrt_major}",
                ],
            }

        from torch_tensorrt import __cuda_version__

        if sys.platform.startswith("win"):
            WIN_LIBS = tensorrt_lib["win"]
            WIN_PATHS = os.environ["PATH"].split(os.path.pathsep)
            for lib in WIN_LIBS:
                ctypes.CDLL(_find_lib(lib, WIN_PATHS))

        elif sys.platform.startswith("linux"):
            LINUX_PATHS = [
                f"/usr/local/cuda-{__cuda_version__}/lib64",
                "/usr/lib",
                "/usr/lib64",
            ]
            if "LD_LIBRARY_PATH" in os.environ:
                LINUX_PATHS += os.environ["LD_LIBRARY_PATH"].split(os.path.pathsep)
            if platform.uname().processor == "x86_64":
                LINUX_PATHS += [
                    "/usr/lib/x86_64-linux-gnu",
                ]
            elif platform.uname().processor == "aarch64":
                LINUX_PATHS += ["/usr/lib/aarch64-linux-gnu"]
            LINUX_LIBS = tensorrt_lib["linux"]
            for lib in LINUX_LIBS:
                ctypes.CDLL(_find_lib(lib, LINUX_PATHS))


alias_tensorrt()
