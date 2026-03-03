from __future__ import annotations

import logging
import sys
from typing import Any, Optional, Tuple

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import tensorrt as trt
import torch
from torch_tensorrt._enums import DeviceType
from torch_tensorrt._features import needs_torch_tensorrt_runtime


class Device(object):
    """
    Defines a device that can be used to specify target devices for engines

    Attributes:
        device_type (torch_tensorrt.DeviceType): Target device type (GPU or DLA). Set implicitly based on if dla_core is specified.
        gpu_id (int): Device ID for target GPU
        dla_core (int): Core ID for target DLA core
        allow_gpu_fallback (bool): Whether falling back to GPU if DLA cannot support an op should be allowed
    """

    device_type: DeviceType = (
        DeviceType.UNKNOWN
    )  #: Target device type (GPU or DLA). Set implicitly based on if dla_core is specified.
    gpu_id: int = -1  #: Device ID for target GPU
    dla_core: int = -1  #: Core ID for target DLA core
    allow_gpu_fallback: bool = (
        False  #: Whether falling back to GPU if DLA cannot support an op should be allowed
    )

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
        if len(args) == 1:
            if not isinstance(args[0], str):
                raise TypeError(
                    "When specifying Device through positional argument, argument must be str"
                )
            else:
                (self.device_type, id) = Device._parse_device_str(args[0])
                if self.device_type == DeviceType.DLA:
                    self.dla_core = id
                    self.gpu_id = 0
                    logging.warning(
                        "Setting GPU id to 0 for device because device 0 manages DLA on AGX Devices",
                    )
                else:
                    self.gpu_id = id

        elif len(args) == 0:
            if "gpu_id" in kwargs or "dla_core" in kwargs:
                if "dla_core" in kwargs:
                    self.dla_core = kwargs["dla_core"]
                if "gpu_id" in kwargs:
                    self.gpu_id = kwargs["gpu_id"]

                if self.dla_core >= 0:
                    self.device_type = DeviceType.DLA
                    if self.gpu_id != 0:
                        self.gpu_id = 0
                        logging.warning(
                            "Setting GPU id to 0 for device because device 0 manages DLA on AGX Platforms",
                        )
                else:
                    self.device_type = DeviceType.GPU
            else:
                raise ValueError(
                    "Either gpu_id or dla_core or both must be defined if no string with device specs is provided as an arg"
                )

        else:
            raise ValueError(
                f"Unexpected number of positional arguments for class Device \n    Found {len(args)} arguments, expected either zero or a single positional arguments"
            )

        if "allow_gpu_fallback" in kwargs:
            if not isinstance(kwargs["allow_gpu_fallback"], bool):
                raise TypeError("allow_gpu_fallback must be a bool")
            self.allow_gpu_fallback = kwargs["allow_gpu_fallback"]

        if "device_type" in kwargs:
            if isinstance(kwargs["device_type"], trt.DeviceType):
                self.device_type = DeviceType._from(kwargs["device_type"])

    def __str__(self) -> str:
        suffix = (
            ")"
            if self.device_type == DeviceType.GPU
            else f", dla_core={self.dla_core}, allow_gpu_fallback={self.allow_gpu_fallback})"
        )
        dev_str: str = f"Device(type={self.device_type}, gpu_id={self.gpu_id}{suffix}"
        return dev_str

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def _from(cls, d: Optional[Self | torch.device | str]) -> Device:
        """Cast a device-type to torch_tensorrt.Device

        Returns the corresponding torch_tensorrt.Device
        """
        if isinstance(d, Device):
            return d

        elif isinstance(d, torch.device):
            if d.type != "cuda":
                raise ValueError('Torch Device specs must have type "cuda"')
            return cls(gpu_id=d.index)

        elif d is None:
            return cls(gpu_id=torch.cuda.current_device())

        else:
            return cls(d)

    @classmethod
    def _from_torch_device(cls, torch_dev: torch.device) -> Device:
        return cls._from(torch_dev)

    @classmethod
    def _current_device(cls) -> Device:
        dev_id = torch.cuda.current_device()
        return cls(gpu_id=dev_id)

    @staticmethod
    def _parse_device_str(s: str) -> Tuple[trt.DeviceType, int]:
        s = s.lower()
        spec = s.split(":")
        if spec[0] == "gpu" or spec[0] == "cuda":
            return (DeviceType.GPU, int(spec[1]))
        elif spec[0] == "dla":
            return (DeviceType.DLA, int(spec[1]))
        else:
            raise ValueError(f"Unknown device type {spec[0]}")

    def to(self, t: type) -> torch.device:
        if t == torch.device:
            if self.gpu_id != -1:
                return torch.device(self.gpu_id)
            else:
                raise ValueError("Invalid GPU ID provided for the CUDA device provided")
        else:
            raise TypeError("Unsupported target type for device conversion")

    @needs_torch_tensorrt_runtime
    def _to_serialized_rt_device(self) -> str:
        delim = torch.ops.tensorrt.SERIALIZED_RT_DEVICE_DELIM()[0]
        dev_info = torch.cuda.get_device_properties(self.gpu_id)
        rt_info = [
            self.gpu_id,
            dev_info.major,
            dev_info.minor,
            int(self.device_type.to(trt.DeviceType)),  # type: ignore[arg-type]
            dev_info.name,
        ]
        rt_info = [str(i) for i in rt_info]
        packed_rt_info: str = delim.join(rt_info)
        logging.debug(f"Serialized Device Info: {packed_rt_info}")
        return packed_rt_info
