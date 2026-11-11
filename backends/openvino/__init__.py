import importlib
from typing import Any

from .partitioner import OpenvinoPartitioner
from .preprocess import OpenvinoBackend

# OpenVINOQuantizer needs nncf, which is only required to quantize, so it is
# imported on first access instead of with the package.
LAZY_IMPORTS = {
    "OpenVINOQuantizer": (
        "executorch.backends.openvino.quantizer.quantizer",
        "OpenVINOQuantizer",
    ),
}

__all__ = ["OpenvinoBackend", "OpenvinoPartitioner"]


def __getattr__(name: str) -> Any:
    if name in LAZY_IMPORTS:
        module_name, attr = LAZY_IMPORTS[name]
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(LAZY_IMPORTS))
