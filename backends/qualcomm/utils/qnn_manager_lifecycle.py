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
from executorch.backends.qualcomm.utils.qnn_sdk_setup import (
    disable_mkldnn_on_amd,
    setup_qnn_sdk,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec

# Thread-local storage for QnnManager instances
_current_qnn_managers = threading.local()


class QnnManagerRegistry:
    def __init__(self):
        self._registry = {}

    def get_or_create_qnn_manager(
        self,
        backend_type: QnnExecuTorchBackendType,
        option: bytes,
        soc_model,
    ) -> PyQnnManager.QnnManager:
        # Outside the branch below, so reusing a cached manager still re-applies them. Both are
        # cheap on a repeat call, and the AMD guard has to hold for every lowering, not only the
        # one that happened to build the manager.
        setup_qnn_sdk()
        disable_mkldnn_on_amd()
        key = (backend_type, soc_model)
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

    def destroy_qnn_manager(
        self, backend_type: QnnExecuTorchBackendType, soc_model
    ):
        key = (backend_type, soc_model)
        if key in self._registry:
            self._registry[key].Destroy()
            del self._registry[key]
        else:
            logging.warning(
                "Attempted to destroy non-existent QnnManager for backend type %s "
                "and SoC %s",
                backend_type.name,
                soc_model.name,
            )


def _get_soc_model(compile_specs: List[CompileSpec]):
    option = generate_qnn_executorch_option(compile_specs)
    python_options = flatbuffer_to_option(option)
    return python_options.soc_info.soc_model


def _get_current_registry() -> QnnManagerRegistry:
    active_registry = getattr(_current_qnn_managers, "active_registry", None)
    if active_registry is None:
        active_registry = QnnManagerRegistry()
        _current_qnn_managers.active_registry = active_registry
    return active_registry


@contextlib.contextmanager
def QnnManagerContext(compile_specs: Dict[str, List[CompileSpec]]):
    current_context_registry = QnnManagerRegistry()
    _current_qnn_managers.active_registry = current_context_registry
    manager_keys_in_this_context = set()
    try:
        for compile_spec_list in compile_specs.values():
            option = generate_qnn_executorch_option(compile_spec_list)
            python_options = flatbuffer_to_option(option)
            backend_type = python_options.backend_options.backend_type
            soc_model = python_options.soc_info.soc_model
            current_context_registry.get_or_create_qnn_manager(
                backend_type, option, soc_model
            )
            manager_keys_in_this_context.add((backend_type, soc_model))
        yield
    finally:
        for backend_type, soc_model in manager_keys_in_this_context:
            current_context_registry.destroy_qnn_manager(backend_type, soc_model)
        _current_qnn_managers.active_registry = None


def get_current_qnn_manager(
    backend_type: QnnExecuTorchBackendType, compile_specs: List[CompileSpec]
) -> PyQnnManager.QnnManager:
    """
    Retrieves the QnnManager instance active for the current QnnManagerContext invocation.
    Return a new QnnManger if no QnnManager is active for the given backend_type in the current context.
    """
    soc_model = _get_soc_model(compile_specs)
    option = generate_qnn_executorch_option(compile_specs)
    # Re-applied even though the manager already exists, because a caller may have turned the
    # setting back on since it was built, and this is a lowering about to run.
    disable_mkldnn_on_amd()
    return _get_current_registry().get_or_create_qnn_manager(
        backend_type, option, soc_model
    )
