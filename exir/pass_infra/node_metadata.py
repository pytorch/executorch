from typing import Any, Dict, Set

from executorch.exir.error import ExportError, ExportErrorType

NodeMetadataValue = Any


PROTECTED_KEYS: Set[str] = {
    "val",
    "stack_trace",
    "nn_module_stack",
    "debug_handle",
}


class NodeMetadata:
    def __init__(self, data: Dict[str, Any]) -> None:
        self.data: Dict[str, Any] = data.copy()

    def __getitem__(self, key: str) -> NodeMetadataValue:
        return self.data[key]

    def __setitem__(self, key: str, value: NodeMetadataValue) -> NodeMetadataValue:
        if key in PROTECTED_KEYS:
            raise ExportError(
                ExportErrorType.VIOLATION_OF_SPEC, f"Could not override node key: {key}"
            )
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def copy(self) -> "NodeMetadata":
        return NodeMetadata(self.data.copy())
