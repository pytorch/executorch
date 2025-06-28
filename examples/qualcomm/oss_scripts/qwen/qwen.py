# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import getpass
import json
import logging
import os
import tempfile
from functools import partial
from multiprocessing.connection import Client

import torch
from executorch.backends.qualcomm._passes import TagQuantIO

from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_capture_program_passes,
)
from executorch.backends.qualcomm.builders.utils import is_graph_output
from executorch.backends.qualcomm.quantizer.custom_annotation import (
    annotate_matmul_16a8w,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.utils.constants import (
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
)

from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.examples.qualcomm.oss_scripts.llama.llama import smart_mask_updater
from executorch.examples.qualcomm.oss_scripts.qwen.model.static_qwen import (
    DecoderModelForCausalLM,
)
from executorch.examples.qualcomm.utils import (
    make_output_dir,
    make_quantizer,
    parse_skip_delegation_node,
    setup_common_args_and_variables,
    SimpleADB,
)
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

HUGGING_FACE_REPO_IDS = {
    "qwen2_5": "Qwen/Qwen2.5-0.5B",
    # "qwen3": "Qwen/Qwen3-0.6B" #TODO: enable this
}

PTE_FILENAME = "static_qwen_qnn_q16"


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

    atten_mask_shape = {
        (
            config.max_batch_size,
            config.ar_len,
            config.max_seq_len,
        ),
    }

    logit_out_shape = {
        (
            config.max_batch_size,
            config.ar_len,
            config.vocab_size,
        )
    }

    freq_shape = {
        (config.ar_len, config.num_key_value_heads // 2),
    }

    freq_op = {
        exir_ops.edge.aten.select.int,
    }
    quant_io_type = None

    if node.op == "placeholder":
        if (
            len(users := list(node.users)) == 1
            and users[0].meta["val"].size()[-2:] in kv_cache_shape
        ):
            quant_io_type = fixed_point_type["kv_type"]
        elif node.meta["val"].size() in logit_out_shape:
            quant_io_type = fixed_point_type["io_type"]
        elif node.meta["val"].size() in atten_mask_shape:
            quant_io_type = fixed_point_type["io_type"]
    if is_graph_output(node):
        if node.meta["val"].size()[-2:] in kv_cache_shape:
            quant_io_type = fixed_point_type["kv_type"]
        elif node.meta["val"].size() in logit_out_shape:
            quant_io_type = fixed_point_type["io_type"]

    # Tag sharding io
    if exir_ops.edge.llama.fallback.default in [
        u.target for u in list(node.users.keys())
    ] + [node.target]:
        quant_io_type = fixed_point_type["io_type"]

    # Tag select op as quantized tensors for freq_sin and freq_cos. It is caused by sharding
    if node.target in freq_op and node.meta["val"].size() in freq_shape:
        quant_io_type = fixed_point_type["io_type"]

    return quant_io_type


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
    model = AutoModelForCausalLM.from_pretrained(model_id).eval()
    config = model.config
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Prepare input
    prompt = args.prompt[0]
    inputs = tokenizer(prompt, return_tensors="pt")

    # Set configs
    config.max_seq_len = args.max_seq_len
    config.ar_len = 1  # kv mode
    config.max_batch_size = 1
    # Some config has head_dim provided that is different from equation below(e.g., qwen3)
    if not hasattr(config, "head_dim"):
        config.head_dim = config.hidden_size // config.num_attention_heads

    # huggingface version
    with torch.no_grad():
        tokens = 1
        past_key_values = StaticCache(
            config=config,
            max_batch_size=config.max_batch_size,
            max_cache_len=config.max_seq_len,
        )
        input_ids = inputs.input_ids
        generated_tokens = input_ids[0].tolist()
        position_ids = torch.tensor([[0]])
        attention_mask = torch.full(
            (1, 1, 1, args.max_seq_len), torch.finfo(torch.float32).min
        )
        while tokens < args.max_seq_len:
            cur_pos = torch.tensor([tokens - 1], dtype=torch.long)
            attention_mask[:, :, :, cur_pos] = 0
            outputs = model(
                input_ids=torch.tensor([generated_tokens[cur_pos]]).unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                num_logits_to_keep=1,
            )

            if tokens >= len(generated_tokens):
                generated_tokens.append(outputs.logits.argmax(dim=-1).item())
            if generated_tokens[-1] == tokenizer.eos_token_id:
                break
            tokens += 1
            position_ids[0, 0] += 1
    logging.info(
        f"Result of Qwen nn.Module with static cache:\n {tokenizer.decode(generated_tokens, skip_special_tokens=True)} \n\n\n"
    )

    # qc static version
    qc_model = DecoderModelForCausalLM(config)
    with tempfile.TemporaryDirectory() as tmp_dir:
        pt_file = f"{tmp_dir}/hf_weights.pt"
        torch.save(model.state_dict(), pt_file)
        qc_model.load_state_dict(torch.load(pt_file, weights_only=True), strict=False)
        qc_model.model.norm.prepare_torch_rms_norm()
        for layer in qc_model.model.layers:
            layer.self_attn.prepare_sha()
            layer.input_layernorm.prepare_torch_rms_norm()
            layer.post_attention_layernorm.prepare_torch_rms_norm()
            layer.mlp.prepare_feedfoward_conv()

    with torch.no_grad():
        _, atten_mask, _, k_caches, v_caches = qc_model.get_example_inputs()
        all_pos = torch.arange(0, args.max_seq_len, 1, dtype=torch.int32).unsqueeze(0)
        token_list = input_ids[0].tolist()
        pos, ar_len = 1, 1
        while token_list[-1] != tokenizer.eos_token_id and pos < args.max_seq_len:
            token = torch.tensor(
                token_list[pos - ar_len : pos], dtype=torch.int32
            ).reshape(1, -1)
            input_pos = all_pos[:, pos - ar_len : pos]
            logits, new_k_caches, new_v_caches = qc_model(
                token,
                atten_mask,
                input_pos,
                *k_caches,
                *v_caches,
            )
            atten_mask, pos, k_caches, v_caches = smart_mask_updater(
                ar_len, atten_mask, pos, k_caches, v_caches, new_k_caches, new_v_caches
            )
            if pos > len(token_list):
                token_list.append(torch.argmax(logits[:, -1], dim=-1).item())
        logging.info(
            f"Result of QC Static Qwen nn.Module:\n {tokenizer.decode(token_list)} \n\n\n"
        )

    with torch.no_grad():
        _, atten_mask, _, k_caches, v_caches = qc_model.get_example_inputs()
        all_pos = torch.arange(0, args.max_seq_len, 1, dtype=torch.int32).unsqueeze(0)
        token_list = input_ids[0].tolist()
        pos, ar_len = 1, 1
        sample_inputs = (
            torch.tensor(token_list[0], dtype=torch.int32).reshape(1, -1),
            atten_mask,
            torch.tensor([[0]], dtype=torch.int32),
            *k_caches,
            *v_caches,
        )
        fx_graph_module = torch.export.export(
            qc_model, sample_inputs, strict=True
        ).module()
        quantizer = make_quantizer(
            quant_dtype=QuantDtype.use_16a8w,
            per_channel_linear=True,
            per_channel_conv=True,
        )
        quantizer.add_custom_quant_annotations(
            (
                partial(
                    annotate_matmul_16a8w,
                    annotate_conv=False,
                ),
            )
        )
        fx_graph_module = prepare_pt2e(fx_graph_module, quantizer)
        print("Calibrating the model...")

        while token_list[-1] != tokenizer.eos_token_id and pos < args.max_seq_len:
            token = torch.tensor(
                token_list[pos - ar_len : pos], dtype=torch.int32
            ).reshape(1, -1)
            input_pos = all_pos[:, pos - ar_len : pos]
            logits, new_k_caches, new_v_caches = fx_graph_module(
                token,
                atten_mask,
                input_pos,
                *k_caches,
                *v_caches,
            )
            atten_mask, pos, k_caches, v_caches = smart_mask_updater(
                ar_len, atten_mask, pos, k_caches, v_caches, new_k_caches, new_v_caches
            )
            if pos > len(token_list):
                token_list.append(torch.argmax(logits[:, -1], dim=-1).item())

        logging.info(f"Result of prepare_pt2e:\n {tokenizer.decode(token_list)} \n\n\n")

        quantized_model = convert_pt2e(fx_graph_module)
        token_list = input_ids[0].tolist()
        pos, ar_len = 1, 1
        while token_list[-1] != tokenizer.eos_token_id and pos < args.max_seq_len:
            token = torch.tensor(
                token_list[pos - ar_len : pos], dtype=torch.int32
            ).reshape(1, -1)
            input_pos = all_pos[:, pos - ar_len : pos]
            logits, new_k_caches, new_v_caches = quantized_model(
                token,
                atten_mask,
                input_pos,
                *k_caches,
                *v_caches,
            )
            atten_mask, pos, k_caches, v_caches = smart_mask_updater(
                ar_len, atten_mask, pos, k_caches, v_caches, new_k_caches, new_v_caches
            )
            if pos > len(token_list):
                token_list.append(torch.argmax(logits[:, -1], dim=-1).item())
        logging.info(f"Result of qdq:\n {tokenizer.decode(token_list)} \n\n\n")

        fixed_point_type = {"kv_type": torch.uint8, "io_type": torch.uint16}
        passes_job = get_capture_program_passes()
        passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
        passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
            "get_quant_io_dtype_fn"
        ] = partial(_tag_ios, fixed_point_type=fixed_point_type, config=config)

        backend_options = generate_htp_compiler_spec(use_fp16=False)
        compiler_spec = generate_qnn_executorch_compiler_spec(
            soc_model=get_soc_to_chipset_map()[args.model],
            backend_options=backend_options,
            shared_buffer=True,
        )

        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            quantized_model,
            sample_inputs,
            compiler_spec,
            constant_methods=qc_model.get_metadata(),
            passes_job=passes_job,
            skip_node_id_set=skip_node_id_set,
            skip_node_op_set=skip_node_op_set,
        )

        executorch_config = ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                alloc_graph_output=False,
            ),
        )
        exec_prog_mgr = edge_prog_mgr.to_executorch(config=executorch_config)
        with open(f"{args.artifact}/{PTE_FILENAME}.pte", "wb") as file:
            exec_prog_mgr.write_to_file(file)


def inference(args):
    workspace = f"/data/local/tmp/{getpass.getuser()}/executorch/{args.artifact}"
    pte_path = f"{args.artifact}/{PTE_FILENAME}.pte"
    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)
    outputs = []

    def post_process():
        with open(f"{args.artifact}/outputs/outputs.txt", "r") as f:
            outputs.append(f.read())

    model_id = HUGGING_FACE_REPO_IDS[args.decoder_model]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer_json = tokenizer.save_pretrained(args.artifact)[-1]
    multi_prompts = " ".join([f'--prompt "{prompt}"' for prompt in args.prompt])
    runner_args = " ".join(
        [
            multi_prompts,
            "--eval_mode 0",
            "--temperature 0",
        ]
    )
    runner_cmd = ""
    performance_output_path = "outputs/inference_speed.txt"
    runner_cmd = " ".join(
        [
            f"cd {workspace} &&",
            "./qnn_llama_runner",
            f"--decoder_model_version {args.decoder_model}",
            "--tokenizer_path tokenizer.json",
            f"--model_path {PTE_FILENAME}.pte",
            f"--seq_len {args.max_seq_len}",
            "--output_path outputs/outputs.txt",
            f"--performance_output_path {performance_output_path}",
            "--kv_updater 'SmartMask'",
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
        runner="examples/qualcomm/oss_scripts/llama/qnn_llama_runner",
    )
    # No pregen inputs, input_list is not required
    adb.push(inputs=[], input_list="", files=[tokenizer_json])
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
        "--pre_gen_pte",
        help="Run the pre-generated Qwen in the given directory.",
        type=str,
    )

    parser.add_argument(
        "--prompt",
        help="User prompts for Qwen. When multiple prompts are entered, a multi-turn conversation will be initiated. Note that this feature is currently for testing purposes only.",
        required=True,
        type=str,
        nargs="+",
    )

    parser.add_argument(
        "--decoder_model",
        choices=["qwen2_5"],
        help="The Qwen model to export. Current available options are: [qwen2_5]",
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
