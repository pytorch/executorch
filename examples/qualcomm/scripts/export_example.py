# pyre-ignore-all-errors
import argparse

import torch
from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.examples.models import MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory
from executorch.extension.export_util.utils import save_pte_program

from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)


def main() -> None:
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

    parser.add_argument(
        "--soc",
        type=str,
        default="SM8650",
        help="Specify the SoC model.",
    )

    parser.add_argument(
        "-q",
        "--quantization",
        choices=["ptq", "qat"],
        help="Run post-traininig quantization.",
    )

    args = parser.parse_args()

    if args.model_name not in MODEL_NAME_TO_MODEL:
        raise RuntimeError(
            f"Model {args.model_name} is not a valid name. "
            f"Available models are {list(MODEL_NAME_TO_MODEL.keys())}."
        )

    # Get model and example inputs
    model, example_inputs, _, _ = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[args.model_name]
    )

    # Get quantizer
    if args.quantization:
        print("Quantizing model...")
        # It is the model quantization path
        quantizer = QnnQuantizer()
        # Typical pytorch 2.0 quantization flow
        m = torch.export.export(model.eval(), example_inputs, strict=True).module()
        if args.quantization == "qat":
            m = prepare_qat_pt2e(m, quantizer)
            # Training loop
            m(*example_inputs)
        elif args.quantization == "ptq":
            m = prepare_pt2e(m, quantizer)
            # Calibration
            m(*example_inputs)
        else:
            raise RuntimeError(f"Unknown quantization type {args.quantization}")
        # Get the quantized model
        m = convert_pt2e(m)
    else:
        # It is the fp model path
        m = model

    # Capture program for edge IR and delegate to QNN backend
    use_fp16 = True if args.quantization is None else False
    backend_options = generate_htp_compiler_spec(
        use_fp16=use_fp16,
    )
    compile_spec = generate_qnn_executorch_compiler_spec(
        soc_model=get_soc_to_chipset_map()[args.soc],
        backend_options=backend_options,
    )
    delegated_program = to_edge_transform_and_lower_to_qnn(
        m, example_inputs, compile_spec, generate_etrecord=args.generate_etrecord
    )

    executorch_program = delegated_program.to_executorch()

    if args.generate_etrecord:
        etrecord_path = args.output_folder + "etrecord.bin"
        executorch_program.get_etrecord().save(etrecord_path)

    save_pte_program(executorch_program, args.model_name, args.output_folder)


if __name__ == "__main__":
    main()  # pragma: no cover
