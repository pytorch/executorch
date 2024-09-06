import argparse
import copy

import torch
from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
from executorch.backends.qualcomm.quantizer.quantizer import (
    get_default_8bit_qnn_ptq_config,
    QnnQuantizer,
)
from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import (
    QcomChipset,
)
from executorch.backends.qualcomm.utils.utils import (
    capture_program,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
)
from executorch.devtools import generate_etrecord
from executorch.examples.models import MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory
from executorch.exir.backend.backend_api import to_backend, validation_disabled
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.extension.export_util.utils import save_pte_program

from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"provide a model name. Valid ones: {list(MODEL_NAME_TO_MODEL.keys())}",
    )
    parser.add_argument(
        "-g",
        "--generate_etrecord",
        action="store_true",
        required=True,
        help="Generate ETRecord metadata to link with runtime results (used for profiling)",
    )

    parser.add_argument(
        "-f",
        "--output_folder",
        type=str,
        default="",
        help="The folder to store the exported program",
    )

    args = parser.parse_args()

    if args.model_name not in MODEL_NAME_TO_MODEL:
        raise RuntimeError(
            f"Model {args.model_name} is not a valid name. "
            f"Available models are {list(MODEL_NAME_TO_MODEL.keys())}."
        )

    model, example_inputs, _ = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[args.model_name]
    )

    # Get quantizer
    quantizer = QnnQuantizer()
    quant_config = get_default_8bit_qnn_ptq_config()
    quantizer.set_bit8_op_quant_config(quant_config)

    # Typical pytorch 2.0 quantization flow
    m = torch.export.export(model.eval(), example_inputs).module()
    m = prepare_pt2e(m, quantizer)
    # Calibration
    m(*example_inputs)
    # Get the quantized model
    m = convert_pt2e(m)

    # Capture program for edge IR
    edge_program = capture_program(m, example_inputs)

    # this is needed for the ETRecord as lowering modifies the graph in-place
    edge_copy = copy.deepcopy(edge_program)

    # Delegate to QNN backend
    backend_options = generate_htp_compiler_spec(
        use_fp16=False,
    )
    qnn_partitioner = QnnPartitioner(
        generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8550,
            backend_options=backend_options,
        )
    )
    with validation_disabled():
        delegated_program = edge_program
        delegated_program.exported_program = to_backend(
            edge_program.exported_program, qnn_partitioner
        )

    executorch_program = delegated_program.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )

    if args.generate_etrecord:
        etrecord_path = args.output_folder + "etrecord.bin"
        generate_etrecord(etrecord_path, edge_copy, executorch_program)

    save_pte_program(executorch_program, args.model_name, args.output_folder)
