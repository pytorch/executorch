import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from executorch.exir._serialize._cord import Cord, CordBuffer
from executorch.exir.tensor_layout import TensorLayout


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

    buffers: Sequence[CordBuffer]
    named_data: Dict[str, DataEntry]


@dataclass
class AlignedData:
    """Data and its required alignment for serialization."""

    data: Cord
    alignment: int

    def __init__(self, data: Cord, alignment: Optional[int] = None) -> None:
        self.data = data
        self.alignment = alignment or 1


def extract_named_data_segments(
    segments: List[AlignedData],
    buffers: Sequence[CordBuffer],
    name_to_data_entry: Dict[str, DataEntry],
) -> Dict[str, int]:
    """Appends unique named-data buffers to segments and returns their indices."""
    segment_index_map: Dict[int, int] = {}
    name_to_segment_index: Dict[str, int] = {}
    for name, data_entry in name_to_data_entry.items():
        alignment = data_entry.alignment or 1
        segment_index = segment_index_map.get(data_entry.buffer_index)
        if segment_index is None:
            segment_index = len(segments)
            segment_index_map[data_entry.buffer_index] = segment_index
            segments.append(
                AlignedData(Cord(buffers[data_entry.buffer_index]), alignment)
            )
        else:
            segments[segment_index].alignment = math.lcm(
                segments[segment_index].alignment, alignment
            )
        name_to_segment_index[name] = segment_index
    return name_to_segment_index


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
