from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Sequence

from executorch.exir._serialize._cord import Cord

from executorch.exir.schema import ScalarType


@dataclass
class TensorLayout:
    """Tensor layout information for externally-serialized tensors.

    Attributes:
        scalar_type: type of the elements in the tensor.
        sizes: size of each dim in the tensor.
        dim_order: specifies the order the dimensions are laid out in memory,
            from outer to inner.
    """

    scalar_type: ScalarType
    sizes: List[int]
    dim_order: List[int]


@dataclass
class TensorEntry:
    """Represents a single tensor in `DataPayload`, specifying its location
    and metadata.

    Attributes:
       buffer_index: The index inside `DataPayload.buffers` that this
            TensorEntry refers to.
       layout: Metadata about the tensor.
    """

    buffer_index: int
    layout: TensorLayout


@dataclass
class DataPayload:
    """Contains the data and metadata required for serialization. Having an
    index-based arrangement instead of Dict[str, bytes] allows the caller to
    deduplicate buffers and point multiple fully qualified names (FQNs) to the
    same entry.

    Attributes:
        buffers: a sequence of tensor buffers.
        fqn_to_buffer: a map from buffer name (fully qualified name) to TensorEntry.
    """

    buffers: Sequence[bytes]
    fqn_to_data: Dict[str, TensorEntry]


class DataSerializer(ABC):
    """Serializes and deserializes FQN-tagged tensor data.

    This base class enables serialization into different formats. See
    executorch/extension/flat_tensor/ for an example.
    """

    @abstractmethod
    def serialize(
        self,
        data: DataPayload,
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
    def deserialize(self, blob: Cord) -> DataPayload:
        """
        Deserializes a blob into a list of tensors. Reverses the effect of
        serialize.

        Args:
            blob: A binary blob that contains the serialized data.

        Returns:
            DataPayload: tensor buffers and tensor layout information
            deserialized from `blob`.
        """
        raise NotImplementedError("deserialize_data")
