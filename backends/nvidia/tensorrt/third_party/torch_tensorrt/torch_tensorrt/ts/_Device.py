import sys
from typing import Any

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import warnings

from torch_tensorrt._Device import Device

try:
    from torch_tensorrt import _C
except ImportError:
    warnings.warn(
        "Unable to import torchscript frontend core and torch-tensorrt runtime. Some dependent features may be unavailable."
    )


class TorchScriptDevice(Device):
    """
    Defines a device that can be used to specify target devices for engines

    Attributes:
        device_type (torch_tensorrt.DeviceType): Target device type (GPU or DLA). Set implicitly based on if dla_core is specified.
        gpu_id (int): Device ID for target GPU
        dla_core (int): Core ID for target DLA core
        allow_gpu_fallback (bool): Whether falling back to GPU if DLA cannot support an op should be allowed
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ Method for torch_tensorrt.Device

        Device accepts one of a few construction patterns

        Args:
            spec (str): String with device spec e.g. "dla:0" for dla, core_id 0

        Keyword Arguments:
            gpu_id (int): ID of target GPU (will get overridden if dla_core is specified to the GPU managing DLA). If specified, no positional arguments should be provided
            dla_core (int): ID of target DLA core. If specified, no positional arguments should be provided.
            allow_gpu_fallback (bool): Allow TensorRT to schedule operations on GPU if they are not supported on DLA (ignored if device type is not DLA)

        Examples:
            - Device("gpu:1")
            - Device("cuda:1")
            - Device("dla:0", allow_gpu_fallback=True)
            - Device(gpu_id=0, dla_core=0, allow_gpu_fallback=True)
            - Device(dla_core=0, allow_gpu_fallback=True)
            - Device(gpu_id=1)
        """
        super().__init__(*args, **kwargs)

    def _to_internal(self) -> _C.Device:
        internal_dev = _C.Device()
        internal_dev.device_type = self.device_type.to(_C.DeviceType)
        internal_dev.gpu_id = self.gpu_id
        internal_dev.dla_core = self.dla_core
        internal_dev.allow_gpu_fallback = self.allow_gpu_fallback
        return internal_dev

    @classmethod
    def _from(cls, d: object) -> Self:
        return cls(
            gpu_id=d.gpu_id,
            dla_core=d.dla_core,
            allow_gpu_fallback=d.allow_gpu_fallback,
        )
