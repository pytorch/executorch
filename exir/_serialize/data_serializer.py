from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Sequence

from executorch.exir._serialize._cord import Cord

from executorch.exir.schema import Tensor


@dataclass
class SerializationInfo:
    # A sequence of deduplicated tensor data.
    tensor_buffers: Sequence[bytes]

    # A map from tensor name (fqn) to tensor index inside `tensor_buffers`.
    fqn_to_buffer_index: Dict[str, int]

    # A map from tensor name (fqn) to tensor metadata. The `Tensor` type contains information related to how tensor data is stored (eg. `data_buffer_idx`, `allocation_info`, `extra_tensor_info`). These are not used; only the tensor metadata (eg. `scalar_type`, `dim_order`, `sizes`, etc.) are used.
    fqn_to_metadata: Dict[str, Tensor]


"""
Abstract base class for custom serialization and deserialization of tensor data from ExecuTorch.
Tensor data can be serialized into different formats. See executorch/extension/flat_tensor/ for an example.
"""


class DataSerializer(ABC):
    @abstractmethod
    def __init__(self) -> None:
        """
        This initializer may be overridden in derived classes to hold
        the data required for serialization, eg. configurations.
        """
        pass

    @abstractmethod
    def serialize_tensors(
        self,
        serialization_info: SerializationInfo,
    ) -> Cord:
        """
        Serializes a list of tensor metadata and tensors emitted by ExecuTorch
        into a binary blob.

        Args:
            serialization_info: the tensor buffers and metadata required for serialization.

        Returns:
            A binary blob that contains the serialized data.
        """
        raise NotImplementedError("serialize_data")

    @abstractmethod
    def deserialize_tensors(self, blob: Cord) -> SerializationInfo:
        """
        Deserializes a blob into a list of tensor metadata and tensors. Reverses the effect of serialize_tensors, which serializes tensor metadata and tensors emitted by ExecuTorch into a binary blob.

        Args:
            blob: A binary blob that contains the serialized data.

        Returns:
            SerializationInfo: tensors and metadata that can be used to serialize data.
        """
        raise NotImplementedError("deserialize_data")
