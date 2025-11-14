from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    PyQnnManagerAdaptor,
)
from executorch.backends.qualcomm.serialization.qc_schema import (
    QcomChipset,
)
from executorch.backends.qualcomm.partition.qnn_partitioner import (
    generate_qnn_executorch_option,
)

dummy_compiler_specs = generate_qnn_executorch_compiler_spec(
    soc_model=QcomChipset.SM8650,
    backend_options=generate_htp_compiler_spec(use_fp16=False),
)
qnn_mgr = PyQnnManagerAdaptor.QnnManager(
    generate_qnn_executorch_option(dummy_compiler_specs)
)
qnn_mgr.Init()
qnn_mgr.Destroy()

qnn_mgr = PyQnnManagerAdaptor.QnnManager(
    generate_qnn_executorch_option(dummy_compiler_specs)
)
qnn_mgr.Init()
qnn_mgr.Destroy()
