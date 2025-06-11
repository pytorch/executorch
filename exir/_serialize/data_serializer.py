from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from executorch.exir._serialize._cord import Cord
from executorch.extension.flat_tensor.serialize.flat_tensor_schema import TensorLayout


@dataclass
class DataEntry:
    """Represents a single blob in `DataPayload`, specifying its location
    and metadata.

    Attributes:
       buffer_index: The index inside `DataPayload.buffers` that this
            DataEntry refers to.
       alignment: The alignment of the data.
       tensor_layout: If this is a tensor, the tensor layout information.
    """

    buffer_index: int
    alignment: int
    tensor_layout: Optional[TensorLayout]


@dataclass
class DataPayload:
    """Contains the data and metadata required for serialization.

    Having an index-based arrangement instead of embedding the buffers in
    DataEntry allows the caller to deduplicate buffers and point multiple
    keys to the same entry.

    Attributes:
        buffers: a sequence of byte buffers.
        key_to_data: a map from unique keys to serializable data.
    """

    buffers: Sequence[bytes]
    named_data: Dict[str, DataEntry]


class DataSerializer(ABC):
    """Serializes and deserializes data. Data can be referenced by a unique key.

    This base class enables serialization into different formats. See
    executorch/extension/flat_tensor/ for an example.
    """

    @abstractmethod
    def serialize(
        self,
        data: DataPayload,
    ) -> Cord:
        """
        Serializes a list of bytes emitted by ExecuTorch into a binary blob.

        Args:
            data: buffers and corresponding metadata used for serialization.


        Returns:
            A binary blob that contains the serialized data.
        """
        raise NotImplementedError("serialize_data")

    @abstractmethod
    def deserialize(self, blob: Cord) -> DataPayload:
        """
        Deserializes a blob into a DataPayload. Reverses the effect of
        serialize.

        Args:
            blob: A binary blob that contains the serialized data.

        Returns:
            DataPayload: buffers and corresponding metadata deserialized
            from `blob`.
        """
        raise NotImplementedError("deserialize_data")
