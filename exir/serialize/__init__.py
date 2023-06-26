# pyre-strict

from executorch.exir.serialize._program import (
    deserialize_from_flatbuffer,
    serialize_to_flatbuffer,
)

__all__ = [
    "deserialize_from_flatbuffer",
    "serialize_to_flatbuffer",
]
