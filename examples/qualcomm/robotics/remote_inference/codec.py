"""
serialization

duty:
PyTorch/NumPy/scalar objects <-> MessagePack bytes.
e.g:
serialization --> MessagePack bytes
MessagePack bytes --> serialization

"""

import msgpack
import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)


class MessagePackTensorCodec:
    """Bounded MessagePack codec supporting PyTorch and NumPy tensors."""

    def __init__(self) -> None:
        pass

    def encode(self, value: object) -> bytes:
        return msgpack.packb(
            value,
            default=self._pack_object,
            use_bin_type=True,
        )

    def decode(self, payload: bytes) -> object:
        return msgpack.unpackb(
            payload,
            object_hook=self._unpack_object,
            raw=False,
        )

    def _pack_object(self, value: object) -> object:
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
            return self._array_to_payload(array, is_torch=True)

        if isinstance(value, np.ndarray):
            return self._array_to_payload(value, is_torch=False)

        if isinstance(value, np.generic):
            return value.item()

        raise TypeError(f"Cannot serialize object of type {type(value)!r}")

    def _unpack_object(self, value: dict) -> object:
        if not value.get("__nd__"):
            return value

        return self._payload_to_tensor(value)

    def _array_to_payload(
        self,
        array: np.ndarray,
        *,
        is_torch: bool,
    ) -> dict[str, object]:
        array = np.ascontiguousarray(array)
        return {
            "__nd__": True,
            "torch": is_torch,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data": array.tobytes(),
        }

    def _payload_to_tensor(
        self,
        payload: dict[str, object],
    ) -> torch.Tensor | np.ndarray:
        if payload.get("__nd__"):
            array = np.frombuffer(payload["data"], dtype=np.dtype(payload["dtype"])).reshape(
                payload["shape"]
            )
            if payload["torch"]:
                return torch.from_numpy(array.copy())
            return array.copy()
        return payload
