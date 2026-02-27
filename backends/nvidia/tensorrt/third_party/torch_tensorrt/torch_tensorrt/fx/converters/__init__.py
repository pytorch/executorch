# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt

if hasattr(trt, "__version__"):
    from .adaptive_avgpool import *  # noqa: F401 F403
    from .add import *  # noqa: F401 F403
    from .batchnorm import *  # noqa: F401 F403
    from .linear import *  # noqa: F401 F403
    from .maxpool import *  # noqa: F401 F403
    from .mul import *  # noqa: F401 F403
    from .transformation import *  # noqa: F401 F403
    from .quantization import *  # noqa: F401 F403
    from .acc_ops_converters import *  # noqa: F401 F403
    from .aten_ops_converters import *  # noqa: F401 F403
    from .nn_ops_converters import *  # noqa: F401 F403

    TRT_LOGGER = trt.Logger()
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
