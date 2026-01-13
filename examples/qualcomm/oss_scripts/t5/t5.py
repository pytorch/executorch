# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import getpass
import json
import os
import subprocess
from multiprocessing.connection import Client

import torch
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    QcomChipset,
    QnnExecuTorchBackendType,
    QnnExecuTorchGpuPrecision,
)
from executorch.backends.qualcomm.utils.utils import (
    generate_gpu_compiler_spec,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.devtools.backend_debug import print_delegation_info
from executorch.examples.qualcomm.oss_scripts.t5.t5_model import (
    CustomT5Stack,
    Seq2SeqLMDecoderExportableModuleWithStaticCache,
    Seq2SeqLMEncoderExportableModule,
    Seq2SeqLMExportableModulePipeline,
)
from executorch.examples.qualcomm.utils import (
    evaluate_squad,
    get_backend_type,
    get_seq2seq_dataset_from_squad_csv,
    make_quantizer,
    replace_module_with_custom_class,
    setup_common_args_and_variables,
    SimpleADB,
)
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.t5.modeling_t5 import T5Stack

PTE_FILE_NAME = "t5_qnn"
ENCODER = "encoder"
DECODER = "decoder"


class T5:
    def __init__(
        self,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        batch_size=1,
        max_hidden_seq_length=4096,
        max_cache_length=1024,
    ):
        self.encoder = (
            Seq2SeqLMEncoderExportableModule(
                model.get_encoder(), max_hidden_seq_length=max_hidden_seq_length
            )
            .to("cpu")
            .eval()
        )
        self.decoder = (
            Seq2SeqLMDecoderExportableModuleWithStaticCache(
                model,
                max_hidden_seq_length=max_hidden_seq_length,
                max_static_cache_length=max_cache_length,
                batch_size=batch_size,
            )
            .to("cpu")
            .eval()
        )

        # Source transformation
        for model in [self.encoder, self.decoder]:
            replace_module_with_custom_class(
                model,
                target_class=T5Stack,
                custom_class=CustomT5Stack,
                extra_custom_kwargs={
                    "max_hidden_seq_length": max_hidden_seq_length,
                    "max_cache_length": max_cache_length,
                },
            )

        # Runner pipeline
        self.pipe = Seq2SeqLMExportableModulePipeline(
            tokenizer,
            model.config,
            max_hidden_seq_length=max_hidden_seq_length,
            max_seq_len=max_cache_length,
        )

        self.exported_encoder = None
        self.exported_decoder = None
        self.quant_dtype = None

    def quantize(self, inputs, quant_dtype, targets=None, metrics=None):
        assert quant_dtype is not None, "quant_dtype must be specified"
        self.quant_dtype = quant_dtype

        with torch.no_grad():

            # Export Modules
            self.exported_encoder = torch.export.export(
                self.encoder, self.encoder.get_example_inputs(), strict=True
            ).module()
            self.exported_decoder = torch.export.export(
                self.decoder, self.decoder.get_example_inputs(), strict=True
            ).module()

            # Quantization
            print(f"Applying quantization with dtype: {quant_dtype}...")
            quantizer = make_quantizer(
                per_channel_linear=True,
                quant_dtype=quant_dtype,
            )

            self.exported_encoder = prepare_pt2e(self.exported_encoder, quantizer)
            self.exported_decoder = prepare_pt2e(self.exported_decoder, quantizer)

            # Calibration
            self.pipe(self.exported_encoder, self.exported_decoder, inputs)

            self.exported_encoder = convert_pt2e(self.exported_encoder)
            self.exported_decoder = convert_pt2e(self.exported_decoder)

            if targets is not None and metrics is not None:
                print(f"Metrics provided for validation: {metrics.__name__}")
                self.pipe.validate(
                    self.exported_encoder,
                    self.exported_decoder,
                    inputs,
                    targets,
                    metrics,
                )
            else:
                print("No targets or metrics provided. Skipping validation step.")

    def lowering_modules(
        self,
        workspace,
        use_fp16=False,
        soc_model=QcomChipset.SM8650,
        backend=QnnExecuTorchBackendType.kHtpBackend,
        skip_node_id_set=None,
        skip_node_op_set=None,
        online_prepare=False,
        verbose=True,
    ):
        graph_names = [ENCODER, DECODER]

        if not self.exported_encoder or not self.exported_decoder:
            modules = [
                self.encoder,
                self.decoder,
            ]
        else:
            modules = [
                self.exported_encoder,
                self.exported_decoder,
            ]

        if backend == QnnExecuTorchBackendType.kGpuBackend and not online_prepare:
            raise RuntimeError("Currently GPU backend only support online_prepare.")
        backend_options = {
            QnnExecuTorchBackendType.kGpuBackend: generate_gpu_compiler_spec(
                **{
                    "precision": (
                        QnnExecuTorchGpuPrecision.kGpuPrecisionFp16
                        if use_fp16
                        else QnnExecuTorchGpuPrecision.kGpuPrecisionUserProvided
                    )
                }
            ),
            QnnExecuTorchBackendType.kHtpBackend: generate_htp_compiler_spec(
                use_fp16=use_fp16
            ),
        }[backend]
        compile_spec = generate_qnn_executorch_compiler_spec(
            soc_model=soc_model,
            backend_options=backend_options,
            online_prepare=online_prepare,
        )
        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            dict(zip(graph_names, modules)),
            dict(
                zip(
                    graph_names,
                    [
                        self.encoder.get_example_inputs(),
                        self.decoder.get_example_inputs(),
                    ],
                )
            ),
            compile_spec,
            constant_methods=self.decoder.get_metadata(),
            skip_node_id_set=skip_node_id_set,
            skip_node_op_set=skip_node_op_set,
            skip_mutable_buffer=False,
        )

        executorch_config = ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=True,
                alloc_graph_output=True,
            ),
            extract_delegate_segments=True,
        )

        if verbose:
            for graph_name in graph_names:
                print_delegation_info(
                    edge_prog_mgr.exported_program(graph_name).graph_module
                )

        exec_prog_mgr = edge_prog_mgr.to_executorch(config=executorch_config)
        with open(f"{workspace}/{PTE_FILE_NAME}.pte", "wb") as file:
            exec_prog_mgr.write_to_file(file)


def main(args):

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    data_size = 100
    max_hidden_seq_length = 384
    max_cache_length = 512

    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small").eval()
    inputs, targets = get_seq2seq_dataset_from_squad_csv(
        args.dataset,
        tokenizer,
        data_size,
        max_hidden_seq_length=max_hidden_seq_length,
        shuffle=False,
    )

    if not args.pre_gen_pte:
        t5 = T5(
            model,
            tokenizer,
            max_hidden_seq_length=max_hidden_seq_length,
            max_cache_length=max_cache_length,
        )
        backend = get_backend_type(args.backend)
        quant_dtype = {
            QnnExecuTorchBackendType.kGpuBackend: None,
            QnnExecuTorchBackendType.kHtpBackend: QuantDtype.use_16a8w,
        }[backend]
        if quant_dtype:
            t5.quantize(inputs, quant_dtype)
        t5.lowering_modules(
            args.artifact,
            soc_model=getattr(QcomChipset, args.model),
            use_fp16=True if quant_dtype is None else False,
            backend=backend,
            online_prepare=args.online_prepare,
        )

    if args.compile_only:
        return

    pte_path = (
        f"{args.pre_gen_pte}/{PTE_FILE_NAME}"
        if args.pre_gen_pte
        else f"{args.artifact}/{PTE_FILE_NAME}"
    ) + ".pte"
    _, _, spiece_model, _, _ = tokenizer.save_pretrained(args.artifact)

    workspace = f"/data/local/tmp/{getpass.getuser()}/executorch/{PTE_FILE_NAME}"

    outputs = []

    def post_process():
        for i in range(len(inputs)):
            with open(f"{args.artifact}/outputs/output_{i}.txt", "r") as f:
                outputs.append(f.read())

    runner_args = " ".join(
        [
            f"--tokenizer_model_path {os.path.basename(spiece_model)}",
            f"--model_path {PTE_FILE_NAME}.pte",
            f"--seq_len {max_cache_length}",
            "--output_folder_path outputs",
        ]
    )
    if args.enable_x86_64:
        # x86 emulator is intended for CI and not performance.
        qnn_sdk = os.getenv("QNN_SDK_ROOT")
        target = "x86_64-linux-clang"
        runner_cmd = " ".join(
            [
                f"export LD_LIBRARY_PATH={qnn_sdk}/lib/{target}/:{args.build_folder}/lib &&",
                f"./{args.build_folder}/examples/qualcomm/oss_scripts/t5/qnn_t5_runner",
                runner_args,
            ]
        )
        subprocess.run(
            runner_cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
        )
        post_process()
    else:
        runner_cmd = " ".join(
            [
                f"cd {workspace} &&",
                "./qnn_t5_runner",
                runner_args,
            ]
        )
        backend = get_backend_type(args.backend)
        adb = SimpleADB(
            qnn_sdk=os.getenv("QNN_SDK_ROOT"),
            build_path=f"{args.build_folder}",
            pte_path=pte_path,
            workspace=workspace,
            device_id=args.device,
            host_id=args.host,
            soc_model=args.model,
            shared_buffer=args.shared_buffer,
            target=args.target,
            runner="examples/qualcomm/oss_scripts/t5/qnn_t5_runner",
            backend=backend,
        )
        adb.push(
            inputs=inputs,
            files=[spiece_model],
        )
        adb.execute(custom_runner_cmd=runner_cmd)
        adb.pull(output_path=args.artifact, callback=post_process)

    result = Seq2SeqLMExportableModulePipeline.evaluate_with_ground_truth(
        tokenizer, outputs, targets, evaluate_squad
    )

    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(json.dumps({"f1": result["f1"]}))
    else:
        print(f"F1 score: {result['f1']}")


if __name__ == "__main__":
    parser = setup_common_args_and_variables()
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts and output by this example. Default ./t5",
        default="./t5",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help=(
            "path to the validation text. "
            "e.g. --dataset SQuAD-v1.1.csv "
            "for https://www.kaggle.com/datasets/akashdesarda/squad-v11?select=SQuAD-v1.1.csv"
        ),
        type=str,
        required=True,
    )

    args = parser.parse_args()
    args.validate(args)
    try:
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
