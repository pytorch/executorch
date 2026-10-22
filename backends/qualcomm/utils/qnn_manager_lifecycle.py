import contextlib
import copy
import logging
import threading
from typing import Dict, List

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
from executorch.backends.qualcomm.partition.utils import generate_qnn_executorch_option
from executorch.backends.qualcomm.serialization.qc_schema import (
    QcomChipset,
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.serialization.qc_schema_serialize import (
    flatbuffer_to_option,
    option_to_flatbuffer,
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
        soc_model=None,
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

    def destroy_all(self):
        for qnn_manager in self._registry.values():
            qnn_manager.Destroy()
        self._registry.clear()


def _get_target_option(
    compile_specs: List[CompileSpec], soc_model: QcomChipset | None
) -> tuple[QnnExecuTorchBackendType, QcomChipset, bytes]:
    option = generate_qnn_executorch_option(compile_specs)
    python_options = flatbuffer_to_option(option)
    target_options = python_options.target_options
    not_fcb = target_options is None
    if not_fcb:
        target_soc_model = python_options.soc_info.soc_model
        if soc_model is not None and soc_model != target_soc_model:
            raise ValueError(f"compile specs do not target {soc_model.name}")
        return python_options.backend_options.backend_type, target_soc_model, option
    for target in target_options.targets:
        if target.soc_info.soc_model == soc_model:
            if len(target.backend_options) != 1:
                raise ValueError(
                    "current multi-SoC execution requires exactly one backend option "
                    f"per target; {soc_model.name} has {len(target.backend_options)}"
                )
            single_target_options = copy.deepcopy(python_options)
            single_target_options.soc_info = target.soc_info
            single_target_options.backend_options = target.backend_options[0]
            return (
                single_target_options.backend_options.backend_type,
                soc_model,
                option_to_flatbuffer(single_target_options),
            )
    if soc_model is None:
        raise ValueError("FCB manager lookup requires soc_model")
    raise ValueError(f"FCB compile specs do not target {soc_model.name}")


@contextlib.contextmanager
def QnnManagerContext(compile_specs: Dict[str, List[CompileSpec]]):
    current_context_registry = QnnManagerRegistry()
    previous_registry = getattr(_current_qnn_managers, "active_registry", None)
    _current_qnn_managers.active_registry = current_context_registry
    try:
        for compile_spec_list in compile_specs.values():
            option = flatbuffer_to_option(
                generate_qnn_executorch_option(compile_spec_list)
            )
            targets = (
                [target.soc_info.soc_model for target in option.target_options.targets]
                if option.target_options is not None
                else [option.soc_info.soc_model]
            )
            for soc_model in targets:
                get_current_qnn_manager(compile_spec_list, soc_model)
        yield
    finally:
        current_context_registry.destroy_all()
        _current_qnn_managers.active_registry = previous_registry


def get_current_qnn_manager(
    compile_specs: List[CompileSpec], soc_model: QcomChipset | None = None
) -> PyQnnManager.QnnManager:
    """
    Retrieves the QnnManager instance active for the current QnnManagerContext invocation.
    Return a new QnnManger if no QnnManager is active for the given backend_type in the current context.
    """
    backend_type, target_soc_model, option = _get_target_option(
        compile_specs, soc_model
    )
    disable_mkldnn_on_amd()
    active_registry = getattr(_current_qnn_managers, "active_registry", None)
    if active_registry is None:
        logging.warning(
            "No QnnManager active for backend type %s  and soc_model %s in the current "
            "QnnManagerContext. It would be better to use "
            "to_edge_transform_and_lower_to_qnn to lower to the QNN backend.",
            backend_type,
            target_soc_model,
        )
        active_registry = QnnManagerRegistry()
    return active_registry.get_or_create_qnn_manager(
        backend_type, option, target_soc_model
    )
