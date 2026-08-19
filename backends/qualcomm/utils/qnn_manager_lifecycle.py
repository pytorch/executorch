import contextlib
import logging
import threading
from typing import Dict, List

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

from executorch.backends.qualcomm.partition.utils import generate_qnn_executorch_option
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
    flatbuffer_to_option,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec

# Thread-local storage for QnnManager instances
_current_qnn_managers = threading.local()


class QnnManagerRegistry:
    def __init__(self):
        # Registry stores {(backend_type, option): QnnManager instance}. Keying on
        # the option bytes as well keeps two specs for one backend -- an fp16
        # partition and a quantized one, say -- from sharing a manager, since
        # precision comes from the options the manager was built with.
        self._registry = {}

    def get_or_create_qnn_manager(
        self, backend_type: QnnExecuTorchBackendType, option: bytes
    ) -> PyQnnManager.QnnManager:
        key = (backend_type, option)
        if key not in self._registry:
            qnn_manager = PyQnnManager.QnnManager(option)
            err = qnn_manager.InitBackend()
            if err.value != 0:
                raise RuntimeError(
                    f"Failed to initialize QNN backend for {backend_type.name}. "
                    "Ensure QNN SDK libraries are available "
                    "(e.g. LD_LIBRARY_PATH includes $QNN_SDK_ROOT/lib/x86_64-linux-clang/)."
                )
            self._registry[key] = qnn_manager
        return self._registry[key]

    def destroy_qnn_manager(self, backend_type: QnnExecuTorchBackendType):
        keys = [k for k in self._registry if k[0] == backend_type]
        if keys:
            for k in keys:
                self._registry[k].Destroy()
                del self._registry[k]
        else:
            logging.warning(
                f"Attempted to destroy non-existent QnnManager for backend type {backend_type.name}"
            )


@contextlib.contextmanager
def QnnManagerContext(compile_specs: Dict[str, List[CompileSpec]]):
    # Create a new registry for the current context
    current_context_registry = QnnManagerRegistry()
    _current_qnn_managers.active_registry = current_context_registry

    backend_types_in_this_context = set()

    try:
        for compile_spec_list in compile_specs.values():
            option = generate_qnn_executorch_option(compile_spec_list)
            python_options = flatbuffer_to_option(option)
            backend_type = python_options.backend_options.backend_type

            # Use the current_context_registry to get/create the manager
            current_context_registry.get_or_create_qnn_manager(backend_type, option)
            backend_types_in_this_context.add(backend_type)
        yield
    finally:
        # Destroy only the managers created within this context
        for backend_type in backend_types_in_this_context:
            current_context_registry.destroy_qnn_manager(backend_type)

        # Clear the active registry reference
        _current_qnn_managers.active_registry = None


def get_current_qnn_manager(
    backend_type: QnnExecuTorchBackendType, compile_specs: List[CompileSpec]
) -> PyQnnManager.QnnManager:
    """
    Retrieves the QnnManager instance active for the current QnnManagerContext invocation.
    Return a new QnnManger if no QnnManager is active for the given backend_type in the current context.
    """
    active_registry = getattr(_current_qnn_managers, "active_registry", None)
    option = generate_qnn_executorch_option(compile_specs)
    if (
        active_registry is None
        or (backend_type, option) not in active_registry._registry
    ):
        logging.warning(
            f"No QnnManager active for backend type {backend_type.name} in the current QnnManagerContext. "
            "It would be better to use to_edge_transform_and_lower_to_qnn to lowering to QNN Backend."
        )
        return QnnManagerRegistry().get_or_create_qnn_manager(backend_type, option)
    return active_registry._registry[(backend_type, option)]
