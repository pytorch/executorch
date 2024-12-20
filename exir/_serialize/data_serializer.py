from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Sequence

from executorch.exir._serialize._cord import Cord

from executorch.exir.schema import ScalarType


@dataclass
class TensorLayout:
    """
    Tensor layout information for externally-serialized tensors.
    """

    scalar_type: ScalarType
    sizes: List[int]
    dim_order: List[bytes]


@dataclass
class SerializationInfo:
    # A sequence of tensor data buffers.
    tensor_buffers: Sequence[bytes]

    # A map from tensor name (fqn) to tensor index inside `tensor_buffers`.
    # Note: multiple tensor names may map to the same index as `tensor_buffers`
    # is likely deduplicated.
    fqn_to_buffer_index: Dict[str, int]

    # A map from tensor name (fqn) to TensorLayout.
    fqn_to_tensor_layout: Dict[str, TensorLayout]


class DataSerializer(ABC):
    """Serializes and deserializes FQN-tagged tensor data.

    This base class enables serialization into different formats. See
    executorch/extension/flat_tensor/ for an example.
    """

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
        Serializes a list of tensors emitted by ExecuTorch into a binary blob.

        Args:
            serialization_info: the tensor buffers and tensor layout
            information required for serialization.

        Returns:
            A binary blob that contains the serialized data.
        """
        raise NotImplementedError("serialize_data")

    @abstractmethod
    def deserialize_tensors(self, blob: Cord) -> SerializationInfo:
        """
        Deserializes a blob into a list of tensors. Reverses the effect of
        serialize_tensors.

        Args:
            blob: A binary blob that contains the serialized data.

        Returns:
            SerializationInfo: tensor buffers and tensor layout information
            deserialized from `blob`.
        """
        raise NotImplementedError("deserialize_data")
