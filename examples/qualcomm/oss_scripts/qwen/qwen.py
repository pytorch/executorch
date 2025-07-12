# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import getpass
import json
import logging
import os
from functools import partial
from multiprocessing.connection import Client

import torch
from executorch.backends.qualcomm._passes import TagQuantIO
from executorch.backends.qualcomm._passes.build_quant_io import BuildQuantIo

from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_capture_program_passes,
)
from executorch.backends.qualcomm.builders.utils import is_graph_output
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.utils.constants import (
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
)
from executorch.devtools.backend_debug import print_delegation_info

from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
    to_edge_transform_and_lower_to_qnn,
    convert_linear_to_conv2d,
)
from torchao.quantization.pt2e import MinMaxObserver

from executorch.examples.qualcomm.utils import (
    make_output_dir,
    make_quantizer,
    parse_skip_delegation_node,
    setup_common_args_and_variables,
    SimpleADB,
)
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoConfig
from executorch.examples.qualcomm.oss_scripts.qwen.decoder_model_wrapper import TorchExportableModuleWithStaticCache
from executorch.examples.qualcomm.oss_scripts.qwen.qwen_model import bypass_rotary_embedding, QCQwen2Attention, initialize_r3_hadamard, replace_qwen2_rms_norm_with_native_rms_norm
from transformers.models.qwen2 import modeling_qwen2
FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

HUGGING_FACE_REPO_IDS = {
    "qwen2.5_0.5B": "Qwen/Qwen2.5-0.5B",
    "qwen2.5_1.5B_instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5_0.5B_instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    # "qwen3": "Qwen/Qwen3-0.6B" #TODO: enable this
}

PTE_FILENAME = "qwen_qnn_q16"


def _tag_ios(node, fixed_point_type, config):

    # shape of k caches and v caches
    kv_cache_shape = {
        # single head, kv input
        (config.head_dim, config.max_seq_len),
        (config.max_seq_len, config.head_dim),
        # single head, kv output
        (config.head_dim, config.ar_len),
        (config.ar_len, config.head_dim),
    }

    logit_out_shape = {
        (
            config.max_batch_size,
            config.ar_len,
            config.vocab_size,
        )
    }

    quant_io_type = None

    if node.op == "placeholder":
        if (
            len(users := list(node.users)) == 1
            and users[0].meta["val"].size()[-2:] in kv_cache_shape
        ):
            quant_io_type = fixed_point_type["kv_type"]
    if is_graph_output(node):
        if node.meta["val"].size()[-2:] in kv_cache_shape:
            quant_io_type = fixed_point_type["kv_type"]
        elif node.meta["val"].size() in logit_out_shape:
            quant_io_type = fixed_point_type["io_type"]

    return quant_io_type

def calibrate(
    graph_module,
    stage,
    max_seq_length,
    tokenizer,
    prompt,
    ):
    with torch.no_grad():
        pos = 0
        inputs = tokenizer(prompt, return_tensors="pt")
        generated_tokens = inputs.input_ids[0].tolist()

        while pos < max_seq_length:
            cur_pos = torch.tensor([pos], dtype=torch.long)
            input_id = torch.tensor([generated_tokens[cur_pos]]).unsqueeze(0)
            outputs = graph_module(
                input_id, cur_pos
            )
            pos += 1
            # graph_module.static_cache.key_cache_0
            if pos >= len(generated_tokens):
                generated_tokens.append(torch.argmax(outputs, dim=-1).item())
            if generated_tokens[-1] == tokenizer.eos_token_id:
                break
    logging.info(
        f"Result of Qwen {stage} with static cache:\n {tokenizer.decode(generated_tokens, skip_special_tokens=True)} \n\n\n"
    )


def compile(args):  # noqa: C901
    skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    if not args.compile_only and args.device is None:
        raise RuntimeError(
            "device serial is required if not compile only. "
            "Please specify a device serial by -s/--device argument."
        )

    model_id = HUGGING_FACE_REPO_IDS[args.decoder_model]
    config =AutoConfig.from_pretrained(model_id)

    device = "cpu"
    batch_size = 1
    dtype = "float32"
    cache_implementation = "static"
    attn_implementation = "eager"

    # Set configs
    config.max_seq_len = args.max_seq_len
    config.ar_len = 1  # kv mode
    config.max_batch_size = batch_size
    # config.num_hidden_layers = 1
    config.enable_spinquant_r3 = True

    # Some config has head_dim provided that is different from equation below(e.g., qwen3)
    if not hasattr(config, "head_dim"):
        config.head_dim = config.hidden_size // config.num_attention_heads
    modeling_qwen2.QWEN2_ATTENTION_CLASSES['eager'] = QCQwen2Attention
    modeling_qwen2.Qwen2RotaryEmbedding.forward = bypass_rotary_embedding
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=dtype,
        config=config,
        attn_implementation=attn_implementation,
        generation_config=GenerationConfig(
            use_cache=True,
            cache_implementation=cache_implementation,
            max_length=args.max_seq_len,
            cache_config={
                "batch_size": batch_size,
                "max_cache_len": args.max_seq_len,
            },
        ),
    ).eval()
    model_wrapper = TorchExportableModuleWithStaticCache(model)
    model_wrapper = initialize_r3_hadamard(model_wrapper)
    model_wrapper = convert_linear_to_conv2d(model_wrapper)
    model_wrapper = replace_qwen2_rms_norm_with_native_rms_norm(model_wrapper)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    with torch.no_grad():
        # Validate on model wrapper
        calibrate(model_wrapper, "nn module", args.max_seq_len, tokenizer, args.prompt)

    use_fp16 = True
    fixed_point_type = {}
    if args.ptq:
        use_fp16 = False
        if args.ptq == "8a8w":
            fixed_point_type["io_type"] = torch.uint8
            fixed_point_type["kv_type"] = torch.uint8
        elif args.ptq in ("16a8w", "16a4w", "16a4w_block", "16a16w",):
            fixed_point_type["io_type"] = torch.uint16
            fixed_point_type["kv_type"] = torch.uint16
        else:
            assert args.ptq in [
                "8a8w",
                "16a8w",
                "16a4w",
                "16a4w_block",
                "16a16w",
            ], f"No support for quant type {args.ptq}. Support 8a8w, 16a4w and 16a4w_block."
        quant_dtype = getattr(QuantDtype, f"use_{args.ptq}")
    fx_graph_module = model_wrapper
    passes_job = get_capture_program_passes()

    with torch.no_grad():
        if not use_fp16:
            fx_graph_module = torch.export.export(
                fx_graph_module,
                args=model_wrapper.get_example_inputs(),
                strict=True,
            ).module()
            quantizer = make_quantizer(
                quant_dtype=quant_dtype,
                per_channel_linear=True,
                per_channel_conv=True,
                act_observer=MinMaxObserver,
            )
            if quant_dtype == QuantDtype.use_16a4w_block:
                conv_nodes = [
                    n for n in fx_graph_module.graph.nodes if "conv" in n.name
                ]
                block_size_map = {n.name: (1, 16, 1, 1) for n in conv_nodes}
                quantizer.set_block_size_map(block_size_map)

            fx_graph_module = prepare_pt2e(fx_graph_module, quantizer)
            calibrate(fx_graph_module,"calibration", args.max_seq_len, tokenizer, args.prompt)

            fx_graph_module = convert_pt2e(fx_graph_module)
            calibrate(fx_graph_module,"qdq model", args.max_seq_len, tokenizer, args.prompt)
            
            passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
            passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
                "get_quant_io_dtype_fn"
            ] = partial(_tag_ios, fixed_point_type=fixed_point_type, config=config)

        backend_options = generate_htp_compiler_spec(use_fp16=use_fp16)
        compiler_spec = generate_qnn_executorch_compiler_spec(
            soc_model=get_soc_to_chipset_map()[args.model],
            backend_options=backend_options,
        )

        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            fx_graph_module,
            model_wrapper.get_example_inputs(),
            compiler_spec,
            constant_methods=model_wrapper.get_metadata(),
            passes_job=passes_job,
            skip_node_id_set=skip_node_id_set,
            skip_node_op_set=skip_node_op_set,
        )

        print_delegation_info(edge_prog_mgr.exported_program().graph_module)
        executorch_config = ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
            ),
             passes=[BuildQuantIo()],
        )
        exec_prog_mgr = edge_prog_mgr.to_executorch(config=executorch_config)
        with open(f"{args.artifact}/{PTE_FILENAME}.pte", "wb") as file:
            exec_prog_mgr.write_to_file(file)
        logging.info(f"Saved exported program to {args.artifact}/{PTE_FILENAME}.pte")

def inference(args):
    workspace = f"/data/local/tmp/{getpass.getuser()}/executorch/{PTE_FILENAME}"
    pte_path = f"{args.artifact}/{PTE_FILENAME}.pte"
    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)
    outputs = []

    def post_process():
        with open(f"{args.artifact}/outputs/result.txt", "r") as f:
            outputs.append(f.read())

    model_id = HUGGING_FACE_REPO_IDS[args.decoder_model]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer_json_path = tokenizer.save_pretrained(args.artifact)[-1]
    # Workaround the error for runtime tokenizer
    # Unsupported behavior 'Isolated' for Split PreTokenizer. Only 'MergedWithPrevious' is supported.
    with open(tokenizer_json_path, "r") as f:
        tokenizer_json = json.load(f)
        tokenizer_json["pre_tokenizer"]["pretokenizers"][0]["behavior"]="MergedWithPrevious"

    with open(tokenizer_json_path, "w") as f:
        json.dump(tokenizer_json, f)
 
    runner_cmd = " ".join(
        [
            f"cd {workspace} &&",
            "./llama_main",
            f'--prompt "{args.prompt}"',
            "--tokenizer_path tokenizer.json",
            f"--model_path {PTE_FILENAME}.pte",
            f"--seq_len {args.max_seq_len}",
            f"--temperature 0",
            " > outputs/result.txt"
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
        runner="examples/models/llama/llama_main",
    )
    # No pregen inputs, input_list is not required
    adb.push(inputs=[], input_list="", files=[tokenizer_json_path])
    adb.execute(custom_runner_cmd=runner_cmd)

    adb.pull(output_path=args.artifact, callback=post_process)

    for idx, output in enumerate(outputs):
        logging.info(f"Results[{idx}]:\n{output}")


def main(args):
    if args.compile_only and args.pre_gen_pte:
        raise RuntimeError("Cannot set both compile_only and pre_gen_pte as true")

    if args.compile_only:
        compile(args)
    elif args.pre_gen_pte:
        inference(args)
    else:
        compile(args)
        inference(args)


if __name__ == "__main__":
    parser = setup_common_args_and_variables()

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example.",
        type=str,
    )

    parser.add_argument(
        "-P",
        "--ptq",
        help="If specified, will do PTQ quantization. default is 16bits activation and 4bits weight. Support 8a8w, 16a4w and 16a4w_block.",
        type=str,
    )

    parser.add_argument(
        "--pre_gen_pte",
        help="Run the pre-generated Qwen in the given directory.",
        type=str,
    )

    parser.add_argument(
        "--prompt",
        help="User prompts for Qwen.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--decoder_model",
        choices=["qwen2.5_0.5B", "qwen2.5_0.5B_instruct", "qwen2.5_1.5B_instruct"],
        help="The Qwen model to export. Current available options are: [qwen2_5, qwen2_5_instruct, qwen2.5_1.5B_instruct]",
        required=True,
    )

    parser.add_argument(
        "--max_seq_len",
        help="This refers to maximum number of tokens that the model can process & consider at once to generate predictions/responses.",
        default=128,
        type=int,
    )

    try:
        args = parser.parse_args()
        if args.artifact is None:
            args.artifact = args.decoder_model
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)