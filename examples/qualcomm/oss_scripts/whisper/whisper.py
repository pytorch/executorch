# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: reenable pyre after fixing the issues
# pyre-ignore-all-errors

import getpass
import json
import logging
import os
import re
import subprocess
from functools import partial
from multiprocessing.connection import Client

import torch
from executorch.backends.qualcomm._passes import TagQuantIO

from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_capture_program_passes,
)
from executorch.backends.qualcomm.builders.utils import is_graph_output

from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils.constants import (
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
)
from executorch.backends.qualcomm.utils.utils import (
    convert_linear_to_conv2d,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
    to_edge_transform_and_lower_to_qnn,
)

from executorch.devtools.backend_debug import print_delegation_info
from executorch.examples.qualcomm.oss_scripts.whisper.whisper_model import (
    QnnSeq2SeqLMDecoderExportableModuleWithStaticCache,
    QnnSeq2SeqLMEncoderExportableModule,
)

from executorch.examples.qualcomm.utils import (
    make_output_dir,
    make_quantizer,
    parse_skip_delegation_node,
    setup_common_args_and_variables,
    SimpleADB,
)
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from torchao.quantization.pt2e import MinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

WHISPER_PTE_FILENAME = "whisper_qnn_16a8w.pte"
ENCODER = "encoder"
DECODER = "decoder"


def get_dataset(data_size):
    from datasets import load_dataset

    dataset = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    processor = AutoProcessor.from_pretrained("openai/whisper-tiny")

    # prepare input data
    inputs, target = [], []
    for index, data in enumerate(dataset):
        if index >= data_size:
            break
        sample = data["audio"]
        feature = processor(
            sample["array"],
            return_tensors="pt",
            truncation=False,
            sampling_rate=sample["sampling_rate"],
        ).input_features
        inputs.append((feature,))
        target.append(data["text"])

    return inputs, target


def calibrate(
    max_seq_length,
    tokenizer,
    whisper_decoder,
    fx_graph_module_encoder,
    fx_graph_module_decoder,
    calibration_inputs,
    decoder_start_token_id=50258,
    eos_token_id=50257,
):
    for i, calibration_input in enumerate(calibration_inputs):
        generated_ids = []
        encoder_output = fx_graph_module_encoder(*calibration_input)
        decoder_input_ids = torch.tensor([[decoder_start_token_id]], dtype=torch.long)
        _, atten_mask, _, _ = whisper_decoder.get_example_inputs()

        # Generate tokens one by one
        for j in range(max_seq_length - 1):
            atten_mask[:, :, :, j] = 0
            # Run decoder for next token prediction
            logits = fx_graph_module_decoder(
                decoder_input_ids,
                atten_mask,
                encoder_output,
                torch.tensor([j], dtype=torch.long),
            )
            # Get next token
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            generated_ids.append(next_token)
            # Update input for next iteration
            decoder_input_ids = torch.tensor([[next_token]], dtype=torch.long)
            # Check if EOS token
            if next_token == eos_token_id:
                break
        # skip_special_tokens=False to align with the results of runner
        logging.info(
            f"Generated result for {i} calibration: {tokenizer.decode(generated_ids, skip_special_tokens=False)}"
        )


def eval_metric(preds, target_strs):
    from torchmetrics.text import WordErrorRate

    def clean_text(rgx_list, text):
        new_text = text
        for rgx_match in rgx_list:
            new_text = re.sub(rgx_match, "", new_text)
        return new_text

    special_strs = ["<|en|>", "<|transcribe|>", "<|notimestamps|>", "<|endoftext|>"]
    special_strs_escape = [re.escape(special_str) for special_str in special_strs]
    pred_str = [clean_text(special_strs_escape, pred).upper() for pred in preds]

    wer = WordErrorRate()
    return wer(pred_str, target_strs)


class Whisper:
    def __init__(
        self, whisper_model, batch_size=1, max_cache_length=1024, max_seq_length=None
    ):
        if max_seq_length is None:
            # Default to max_cache_size if max_seq_len is not specified
            self.max_seq_length = max_cache_length
        elif max_seq_length > max_cache_length:
            logging.warning(
                f"max_seq_length={max_seq_length} is larger than max_cache_length={max_cache_length}. Generating tokens will be truncated to max_cache_length."
            )
            self.max_seq_length = max_cache_length
        else:
            self.max_seq_length = max_seq_length
        self.whisper_model = whisper_model
        self.config = whisper_model.config
        self.head_dim = (
            self.config.head_dim
            if hasattr(self.config, "head_dim")
            else self.config.hidden_size // self.config.num_attention_heads
        )

        self.whisper_encoder = (
            QnnSeq2SeqLMEncoderExportableModule(whisper_model.get_encoder())
            .to("cpu")
            .eval()
        )
        self.encoder_passes_job = get_capture_program_passes()

        self.whisper_decoder = (
            QnnSeq2SeqLMDecoderExportableModuleWithStaticCache(
                whisper_model=whisper_model,
                max_cache_length=self.max_seq_length,
                batch_size=batch_size,
            )
            .to("cpu")
            .eval()
        )
        # To improve the performance
        self.whisper_decoder = convert_linear_to_conv2d(self.whisper_decoder)
        self.decoder_passes_job = get_capture_program_passes()
        self.exported_whisper_encoder = None
        self.exported_whisper_decoder = None
        self.has_quant_io = False
        self.kv_shape = {
            (self.max_seq_length, self.head_dim),
        }

    def _tag_ios(self, node, fixed_point_type):
        if not self.has_quant_io:
            return

        quant_io_type = None
        if node.op == "placeholder" and node.meta["val"].size()[-2:] in self.kv_shape:
            quant_io_type = fixed_point_type

        if is_graph_output(node):
            # shape of k caches and v caches
            if node.meta["val"].size()[-2:] in self.kv_shape:
                quant_io_type = fixed_point_type

        return quant_io_type

    def quantize(
        self, calibration_inputs, quant_dtype, tokenizer, custom_annotations=()
    ):
        self.quant_dtype = quant_dtype
        self.has_quant_io = True

        # Need to set per_channel_linear=True for encoder to enhance accuracy
        quantizer = make_quantizer(
            quant_dtype=quant_dtype,
            per_channel_conv=True,
            per_channel_linear=True,
            act_observer=MinMaxObserver,
            custom_annotations=custom_annotations,
            eps=2**-20,
        )

        with torch.no_grad():
            self.exported_whisper_encoder = torch.export.export(
                self.whisper_encoder,
                self.whisper_encoder.get_example_inputs(),
                strict=True,
            ).module()
            self.exported_whisper_decoder = torch.export.export(
                self.whisper_decoder,
                self.whisper_decoder.get_example_inputs(),
                strict=True,
            ).module()

            self.exported_whisper_encoder = prepare_pt2e(
                self.exported_whisper_encoder, quantizer
            )
            self.exported_whisper_decoder = prepare_pt2e(
                self.exported_whisper_decoder, quantizer
            )

            logging.info("Quantizing the model...")

            calibrate(
                self.max_seq_length,
                tokenizer,
                self.whisper_decoder,
                self.exported_whisper_encoder,
                self.exported_whisper_decoder,
                calibration_inputs,
                decoder_start_token_id=getattr(
                    self.config, "decoder_start_token_id", None
                ),
                eos_token_id=getattr(self.config, "eos_token_id", None),
            )

            self.exported_whisper_encoder = convert_pt2e(self.exported_whisper_encoder)
            self.exported_whisper_decoder = convert_pt2e(self.exported_whisper_decoder)

            self.decoder_passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
            self.decoder_passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
                "get_quant_io_dtype_fn"
            ] = partial(self._tag_ios, fixed_point_type=torch.uint16)

    def lowering_modules(
        self,
        workspace,
        use_fp16=False,
        soc_model=QcomChipset.SM8650,
        skip_node_id_set=None,
        skip_node_op_set=None,
        verbose=True,
    ):
        logging.info("Lowering the model...")
        executorch_config = ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=True,
                alloc_graph_output=True,
            ),
            extract_delegate_segments=True,
        )
        with torch.no_grad():
            # backend option
            backend_options = generate_htp_compiler_spec(use_fp16=use_fp16)
            compiler_specs = generate_qnn_executorch_compiler_spec(
                soc_model=soc_model,
                backend_options=backend_options,
            )

            whisper_edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
                {
                    ENCODER: self.exported_whisper_encoder,
                    DECODER: self.exported_whisper_decoder,
                },
                {
                    ENCODER: self.whisper_encoder.get_example_inputs(),
                    DECODER: self.whisper_decoder.get_example_inputs(),
                },
                {ENCODER: compiler_specs, DECODER: compiler_specs},
                constant_methods=self.whisper_decoder.get_metadata(),
                passes_job={
                    ENCODER: get_capture_program_passes(),
                    DECODER: self.decoder_passes_job,
                },
                skip_node_id_set=skip_node_id_set,
                skip_node_op_set=skip_node_op_set,
                skip_mutable_buffer=False,
            )

            if verbose:
                print_delegation_info(
                    whisper_edge_prog_mgr.exported_program(ENCODER).graph_module
                )
                print_delegation_info(
                    whisper_edge_prog_mgr.exported_program(DECODER).graph_module
                )
            whisper_edge_prog_mgr = whisper_edge_prog_mgr.to_executorch(
                config=executorch_config
            )
            with open(f"{workspace}/{WHISPER_PTE_FILENAME}", "wb") as file:
                whisper_edge_prog_mgr.write_to_file(file)


def compile_whisper(args, inputs):
    skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("openai/whisper-tiny")
    module = (
        AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny")
        .to("cpu")
        .eval()
    )

    max_cache_length = 1024
    batch_size = 1
    whisper = Whisper(
        module,
        batch_size=batch_size,
        max_cache_length=max_cache_length,
        max_seq_length=args.max_seq_len,
    )

    whisper.quantize(inputs, QuantDtype.use_16a8w, tokenizer)
    whisper.lowering_modules(
        args.artifact,
        use_fp16=False,
        soc_model=get_soc_to_chipset_map()[args.model],
        skip_node_id_set=skip_node_id_set,
        skip_node_op_set=skip_node_op_set,
    )


def inference_whisper(args, inputs, target):
    workspace = f"/data/local/tmp/{getpass.getuser()}/executorch/whisper"
    tokenizer = AutoTokenizer.from_pretrained("openai/whisper-tiny")
    tokenizer_json = tokenizer.save_pretrained(args.artifact)[-1]
    pte_path = (
        f"{args.pre_gen_pte}/{WHISPER_PTE_FILENAME}"
        if args.pre_gen_pte
        else f"{args.artifact}/{WHISPER_PTE_FILENAME}"
    )

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)
    outputs = []

    def post_process():
        for i in range(len(inputs)):
            with open(f"{args.artifact}/outputs/output_{i}.txt", "r") as f:
                outputs.append(f.read())

    seq_len = args.max_seq_len
    runner_args = " ".join(
        [
            f"--model_path {WHISPER_PTE_FILENAME}",
            f"--tokenizer_json_path {os.path.basename(tokenizer_json)}",
            "--input_list_path input_list.txt",
            f"--seq_len {seq_len}",
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
                f"./{args.build_folder}/examples/qualcomm/oss_scripts/whisper/qnn_whisper_runner",
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
                "./qnn_whisper_runner",
                runner_args,
            ]
        )

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
            runner="examples/qualcomm/oss_scripts/whisper/qnn_whisper_runner",
        )
        # No pregen inputs, input_list is not required
        adb.push(inputs=inputs, files=[tokenizer_json])
        adb.execute(custom_runner_cmd=runner_cmd)

        adb.pull(output_path=args.artifact, callback=post_process)
    wer = eval_metric(outputs, target)

    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(
                json.dumps(
                    {
                        "wer": float(wer),
                    }
                )
            )
    else:
        logging.info(f"Wer: {wer}")
        for idx, output in enumerate(outputs):
            logging.info(f"Results[{idx}]:\n{output}")


if __name__ == "__main__":
    parser = setup_common_args_and_variables()

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. "
        "Default ./whisper",
        default="./whisper",
        type=str,
    )

    parser.add_argument(
        "--max_seq_len",
        help="Maximum sequence length for the generated output.  Defaults to use the model's `max_cache_size` attribute. Will be truncated to maximal cache size if larger than `max_cache_size`.",
        default=1024,
        type=int,
    )

    args = parser.parse_args()
    args.validate(args)

    if args.compile_only and args.pre_gen_pte:
        exit("Cannot set both compile_only and pre_gen_pte as true")

    data_num = 20
    if args.ci:
        inputs = [(torch.rand(1, 80, 3000),)]
        logging.warning(
            "This option is for CI to verify the export flow. It uses random input and will result in poor accuracy."
        )
    else:
        inputs, target = get_dataset(data_num)

    if args.pre_gen_pte:
        inference_whisper(args, inputs, target)
        exit(f"Finish the running pre_gen_pte from {args.pre_gen_pte}")

    if args.compile_only:
        compile_whisper(args, inputs)
        exit(f"Finish compile_only and save to {args.artifact}")

    try:
        compile_whisper(args, inputs)
        inference_whisper(args, inputs, target)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
