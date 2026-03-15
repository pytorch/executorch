import logging

from torch_tensorrt._utils import sanitized_torch_version

from packaging import version

logger = logging.getLogger(__name__)

if version.parse(sanitized_torch_version()) >= version.parse("2.1.dev"):
    from ._compiler import (
        compile,
        convert_exported_program_to_serialized_trt_engine,
        cross_compile_for_windows,
        load_cross_compiled_exported_program,
        save_cross_compiled_exported_program,
    )
    from ._exporter import export
    from ._refit import refit_module_weights
    from ._settings import CompilationSettings
    from ._SourceIR import SourceIR
    from ._tracer import trace
    from .debug._Debugger import Debugger
