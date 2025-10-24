import os

import torch

from executorch.backends.qualcomm.tests.models import SimpleModel
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    QcomChipset,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.devtools import Inspector
from executorch.devtools.inspector._inspector_utils import TimeScale
from executorch.examples.qualcomm.utils import (
    make_quantizer,
    setup_common_args_and_variables,
    SimpleADB,
)

from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


def main(args):
    # capture nn.Module into ExportedProgram
    sample_input = (torch.randn(1, 32, 28, 28), torch.randn(1, 32, 28, 28))
    model = torch.export.export(SimpleModel(), sample_input).module()

    pte_filename = "qnn_simple_model"

    # Quantize the model
    quantizer = make_quantizer()
    prepared = prepare_pt2e(model, quantizer)
    prepared(*sample_input)
    converted = convert_pt2e(prepared)

    # setup compile spec for HTP backend
    backend_options = generate_htp_compiler_spec(use_fp16=False)
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=QcomChipset.SM8750,
        backend_options=backend_options,
        profile=True,
    )
    # lower to QNN ExecuTorch Backend
    edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
        module=converted,
        inputs=sample_input,
        compiler_specs=compiler_specs,
        generate_etrecord=True,
    )

    # store pte file
    exec_prog = edge_prog_mgr.to_executorch()
    with open(f"{pte_filename}.pte", "wb") as f:
        exec_prog.write_to_file(f)

    # setup ADB for on-device execution
    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        build_path=f"{args.build_folder}",
        pte_path=f"{pte_filename}.pte",
        workspace=f"/data/local/tmp/executorch/{pte_filename}",
        device_id=args.device,
        soc_model=args.model,
        target=args.target,
    )
    adb.push(inputs=[sample_input])
    adb.execute()

    # pull etdump back and display the statistics
    adb.pull_etdump(".")
    exec_prog.get_etrecord().save("etrecord.bin")
    inspector = Inspector(
        etdump_path="etdump.etdp",
        etrecord="etrecord.bin",
        source_time_scale=TimeScale.CYCLES,
        target_time_scale=TimeScale.CYCLES,
    )
    df = inspector.to_dataframe()
    # here we only dump the first 15 rows
    if args.num_rows > 0:
        df = df.head(args.num_rows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    parser = setup_common_args_and_variables()
    parser.add_argument(
        "--num_rows",
        type=int,
        default=-1,
        help="The number of rows for etdump",
    )

    args = parser.parse_args()
    main(args)
