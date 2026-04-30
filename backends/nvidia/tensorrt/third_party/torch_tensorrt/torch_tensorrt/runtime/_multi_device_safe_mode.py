import logging
from typing import Any

import torch
import torch_tensorrt

if torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime:
    _PY_RT_MULTI_DEVICE_SAFE_MODE = torch.ops.tensorrt.get_multi_device_safe_mode()
else:
    _PY_RT_MULTI_DEVICE_SAFE_MODE = False


logger = logging.getLogger(__name__)


class _MultiDeviceSafeModeContextManager(object):
    """Helper class used in conjunction with `set_multi_device_safe_mode`

    Used to enable `set_multi_device_safe_mode` as a dual-purpose context manager
    """

    def __init__(self, old_mode: bool) -> None:
        self.old_mode = old_mode

    def __enter__(self) -> "_MultiDeviceSafeModeContextManager":
        return self

    def __exit__(self, *args: Any) -> None:
        # Set multi-device safe mode back to old mode in Python
        global _PY_RT_MULTI_DEVICE_SAFE_MODE
        _PY_RT_MULTI_DEVICE_SAFE_MODE = self.old_mode

        # Set multi-device safe mode back to old mode in C++
        if torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime:
            torch.ops.tensorrt.set_multi_device_safe_mode(self.old_mode)


def set_multi_device_safe_mode(mode: bool) -> _MultiDeviceSafeModeContextManager:
    """Sets the runtime (Python-only and default) into multi-device safe mode

    In the case that multiple devices are available on the system, in order for the
    runtime to execute safely, additional device checks are necessary. These checks
    can have a performance impact so they are therefore opt-in. Used to suppress
    the warning about running unsafely in a multi-device context.

    Arguments:
        mode (bool): Enable (``True``) or disable (``False``) multi-device checks

    Example:

        .. code-block:: py

            with torch_tensorrt.runtime.set_multi_device_safe_mode(True):
                results = trt_compiled_module(*inputs)

    """
    # Fetch existing safe mode and set new mode for Python
    global _PY_RT_MULTI_DEVICE_SAFE_MODE
    old_mode = _PY_RT_MULTI_DEVICE_SAFE_MODE
    _PY_RT_MULTI_DEVICE_SAFE_MODE = mode

    # Set new mode for C++
    if torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime:
        torch.ops.tensorrt.set_multi_device_safe_mode(mode)

    logger.info(f"Set multi-device safe mode to {mode}")

    # Return context manager in case the function is used in a `with` call
    return _MultiDeviceSafeModeContextManager(old_mode)
