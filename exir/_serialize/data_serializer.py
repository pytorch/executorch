from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

from executorch.exir._serialize._cord import Cord

from executorch.exir.schema import Tensor


# Abstract base class that data serializers should adhere to.
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
        tensor_buffer: List[bytes],
        tensor_map: Dict[str, int],
        tensor_metadata: Dict[str, Tensor],
    ) -> Union[Cord, bytes, bytearray]:
        """
        Serializes a list of tensor metadata and tensors emitted by ExecuTorch
        into a binary blob.

        Args:
            tensor_buffer: A list of deduplicated tensor data.
            tensor_map: A map from tensor name (fqn) to tensor index inside 'tensor_buffer'.
            tensor_metadata: A map from tensor name (fqn) to tensor metadata.

        Returns:
            A binary blob that contains the serialized data.
        """
        raise NotImplementedError("serialize_data")

    @abstractmethod
    def deserialize_tensors(
        self, blob: Union[Cord, bytes, bytearray]
    ) -> Tuple[List[bytes], Dict[str, int], Dict[str, Tensor]]:
        """
        Deserializes a blob into a list of tensor metadata and tensors. Reverses the effect of serialize_tensors.

        Args:
            blob: A binary blob that contains the serialized data.

        Returns:
            A tuple of (tensor_buffer, tensor_map, tensor_metadata).
        """
        raise NotImplementedError("deserialize_data")
