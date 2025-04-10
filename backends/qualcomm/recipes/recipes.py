# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
from typing import Optional

from executorch import exir
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype

from executorch.examples.qualcomm.utils import make_quantizer
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, ExportRecipe
from executorch.exir.passes import MemoryPlanningPass
from torch.ao.quantization.quantizer import Quantizer

def get_qualcomm_htp_et_recipe(
    name: str,
    soc_model: str = "SM8650",
    qnn_version: str = "2.25",
    quant_dtype: Optional[QuantDtype] = None,
    shared_buffer=False,
    skip_node_id_set: set[int] = set(),
    skip_node_op_set: set[str] = set(),
) -> ExportRecipe:
    from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset

    from executorch.backends.qualcomm.utils.utils import (
        _transform,
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
    )

    if quant_dtype:
        qnn_quantizer: Quantizer  = make_quantizer(quant_dtype=quant_dtype)
    else:
        qnn_quantizer = make_quantizer()

    qnn_partitioner = QnnPartitioner(
        generate_qnn_executorch_compiler_spec(
            soc_model=getattr(QcomChipset, soc_model),
            backend_options=generate_htp_compiler_spec(
                use_fp16=False if quant_dtype else True
            ),
        ),
        skip_node_id_set=skip_node_id_set,
        skip_node_op_set=skip_node_op_set,
    )
    
    executorch_config = ExecutorchBackendConfig(
        # For shared buffer, user must pass the memory address
        # which is allocated by RPC memory to executor runner.
        # Therefore, won't want to pre-allocate
        # by memory manager in runtime.
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=not shared_buffer,
            alloc_graph_output=not shared_buffer,
        ),
    )

    return ExportRecipe(
        name,
        quantizer=qnn_quantizer,
        partitioners=[qnn_partitioner],
        pre_edge_transform_passes=_transform,
        edge_compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
        edge_transform_passes=[],
        transform_check_ir_validity=True,
        executorch_backend_config=executorch_config,
    )
