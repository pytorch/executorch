# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: reenable pyre after fixing the issues
# pyre-ignore-all-errors

import codecs
import getpass
import json
import os
import time
from multiprocessing.connection import Client

import torch
from executorch.backends.qualcomm._passes.build_quant_io import BuildQuantIo

from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner

from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils.constants import QCOM_QUANTIZED_IO
from executorch.backends.qualcomm.utils.utils import (
    capture_program,
    convert_linear_to_conv2d,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
)
from executorch.examples.qualcomm.oss_scripts.llama2.model.static_llama import (
    LlamaModel,
    ModelArgs,
)
from executorch.examples.qualcomm.utils import (
    make_output_dir,
    make_quantizer,
    setup_common_args_and_variables,
    SimpleADB,
)
from executorch.exir import EdgeCompileConfig, EdgeProgramManager
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.extension.llm.export.builder import DType

from sentencepiece import SentencePieceProcessor
from torch.ao.quantization.observer import MinMaxObserver
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e


pte_filename = "llama2_qnn"


def annotate_matmul_16a8w(gm: torch.fx.GraphModule) -> None:
    """
    This function is specific for matmul op 16a8w.
    """

    from executorch.backends.qualcomm.quantizer.annotators import QUANT_ANNOTATION_KEY
    from executorch.backends.qualcomm.quantizer.quantizer import (
        get_16a8w_qnn_ptq_config,
        get_8a8w_qnn_ptq_config,
        QuantizationConfig,
    )
    from torch.ao.quantization.quantizer import (
        QuantizationAnnotation,
        SharedQuantizationSpec,
    )
    from torch.fx import Node

    def annotate_matmul(node: Node, quantization_config: QuantizationConfig):
        input_qspec_map = {}
        input_act = node.args[0]
        input_spec = quantization_config.input_activation
        input_qspec_map[input_act] = input_spec

        input_act1 = node.args[1]
        input_spec1 = quantization_config.weight
        input_qspec_map[input_act1] = input_spec1

        node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )

    def annotate_cat(node: Node, quantization_config: QuantizationConfig):
        input_nodes = node.args[0]

        first_input_node = input_nodes[0]
        input_qspec_map = {}
        input_qspec_map[first_input_node] = quantization_config.input_activation
        share_qparams_with_input_act0_qspec = SharedQuantizationSpec(
            (first_input_node, node)
        )

        for input_node in input_nodes[1:]:
            if input_node not in input_qspec_map:
                input_qspec_map[input_node] = share_qparams_with_input_act0_qspec

        node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=share_qparams_with_input_act0_qspec,
            _annotated=True,
        )

    def annotate_single_in_single_out(
        node: Node, quantization_config: QuantizationConfig
    ) -> None:

        input_qspec_map = {}
        input_act = node.args[0]
        input_qspec_map[input_act] = quantization_config.input_activation

        node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )

    def annotate_matmul_input1(node: Node):
        quantization_config_8a8w = get_8a8w_qnn_ptq_config(act_symmetric=True)
        while isinstance(node, Node) and node.op == "call_function":
            if node.target in [
                torch.ops.aten.permute.default,
                torch.ops.aten.transpose.int,
            ]:
                annotate_single_in_single_out(node, quantization_config_8a8w)
                node = node.args[0]
            elif node.target == torch.ops.aten.cat.default:
                annotate_cat(node, quantization_config_8a8w)
                node = node.args[0][0]
            else:
                node = node.args[0]

    quantization_config_16a8w = get_16a8w_qnn_ptq_config()

    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.matmul.default:
            annotate_matmul(node, quantization_config_16a8w)
            annotate_matmul_input1(node.args[1])


def annotate_linear_16a8w_in_affine_layer(gm: torch.fx.GraphModule) -> None:
    from executorch.backends.qualcomm.quantizer.annotators import QUANT_ANNOTATION_KEY
    from executorch.backends.qualcomm.quantizer.quantizer import (
        get_ptq_per_channel_quant_config,
        QuantizationConfig,
    )
    from torch.ao.quantization.quantizer import QuantizationAnnotation
    from torch.fx import Node

    def annotate_conv2d(node: Node, quantization_config: QuantizationConfig) -> None:
        input_qspec_map = {}
        input_act = node.args[0]
        input_spec = quantization_config.input_activation
        input_qspec_map[input_act] = input_spec

        weight = node.args[1]
        input_qspec_map[weight] = quantization_config.weight

        node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )

    quantization_config_16a8w_per_channel = get_ptq_per_channel_quant_config(
        torch.uint16, weight_dtype=torch.int8
    )
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.conv2d.default:
            if "nn_module_stack" in node.meta:
                module_values_list = list(node.meta["nn_module_stack"].values())
                full_qualified_name = module_values_list[0][0]
                if full_qualified_name == "L['self'].llama.output":
                    annotate_conv2d(
                        node, quantization_config=quantization_config_16a8w_per_channel
                    )


def _kv_calibrate(
    example_inputs,
    user_prompts,
    module: torch.fx.GraphModule,
    tokenizer_model_path="tokenizer.model",
    max_seq_len=512,
):
    sp_model = SentencePieceProcessor(model_file=tokenizer_model_path)
    _, atten_mask, _, k_caches, v_caches = example_inputs

    # TODO: change criteria & support batch inputs if necessary
    pos = torch.tensor(0, dtype=torch.int32)
    token_list = [sp_model.bos_id()]
    for prompt in user_prompts.split():
        token_list += sp_model.encode(prompt)

    def sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
        probs_sort, probs_indices = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0
        probs_sort /= probs_sort.sum(dim=-1, keepdim=True)
        next_token = torch.multinomial(probs_sort, num_samples=1)
        return probs_indices.gather(dim=-1, index=next_token)

    with torch.no_grad():
        while token_list[-1] != sp_model.eos_id() and pos < max_seq_len - 1:
            logits, new_k_caches, new_v_caches = module(
                torch.full((1, 1), token_list[pos]),
                atten_mask,
                torch.full((1, 1), pos),
                *k_caches,
                *v_caches,
            )
            k_caches = [
                torch.cat([k_cache[:, :, 1:], new_k_caches[i]], dim=-1)
                for i, k_cache in enumerate(k_caches)
            ]
            v_caches = [
                torch.cat([v_cache[:, 1:, :], new_v_caches[i]], dim=1)
                for i, v_cache in enumerate(v_caches)
            ]

            pos += 1
            atten_mask[0][-pos - 1] = 0
            if pos >= len(token_list):
                probs = torch.softmax(logits[:, -1] / 0.8, dim=-1)
                token_list.append(sample_top_p(probs, 0.9).item())

    print(f"calibration data:\n{sp_model.decode(token_list)}")


def _batch_prefill_calibrate(
    example_inputs,
    user_prompts,
    module: torch.fx.GraphModule,
    tokenizer_model_path="tokenizer.model",
    max_seq_len=512,
):
    sp_model = SentencePieceProcessor(model_file=tokenizer_model_path)
    _, atten_mask = example_inputs
    max_cache_len = max_seq_len - 1

    # TODO: change criteria & support batch inputs if necessary
    token_list = sp_model.encode(user_prompts, bos=True, eos=False)
    token_list = torch.tensor(token_list)[:max_cache_len].reshape(1, -1)
    last_prompt_pos = token_list.numel()
    if last_prompt_pos < max_cache_len:
        token_list = torch.cat(
            [
                token_list,
                torch.zeros((1, max_cache_len - last_prompt_pos), dtype=torch.int32),
            ],
            dim=1,
        )
    else:
        token_list = token_list[:, :max_cache_len]

    with torch.no_grad():
        logits, new_k_caches, new_v_caches = module(
            token_list,
            atten_mask,
        )
        predict = [torch.argmax(logits[:, last_prompt_pos - 1], dim=-1).item()]

    print(f"calibration data:\n{sp_model.decode(predict)}")


def calibrate(
    example_inputs,
    user_prompts,
    module: torch.fx.GraphModule,
    tokenizer_model_path="tokenizer.model",
    max_seq_len=512,
):
    if len(example_inputs) == 2:
        _batch_prefill_calibrate(
            example_inputs,
            user_prompts,
            module,
            tokenizer_model_path,
            max_seq_len,
        )
    elif len(example_inputs) == 5:
        _kv_calibrate(
            example_inputs,
            user_prompts,
            module,
            tokenizer_model_path,
            max_seq_len,
        )
    else:
        raise RuntimeError("Get wrong inputs")


class SingleLlama:
    def __init__(self, llama_model) -> None:
        super().__init__()
        self.llama_model = llama_model
        self.quant_dtype = None
        self.llama_meta = self.llama_model.get_metadata()
        self.has_quant_io = False
        if self.llama_meta["get_use_kv_cache"]:
            tokens, atten_mask, pos_ids, k_caches, v_caches = self.get_example_inputs(
                use_kv_cache=True
            )
            self.inputs = (tokens, atten_mask, pos_ids, *k_caches, *v_caches)
        else:
            tokens, atten_mask = self.get_example_inputs(use_kv_cache=False)
            self.inputs = (tokens, atten_mask)

    def _tag_kv_ios(self, gm: torch.fx.GraphModule, kv_type):
        if not self.has_quant_io:
            return

        # shape of k caches and v caches
        input_cache_shape = {
            (self.llama_meta["get_head_dim"], self.llama_meta["get_max_seq_len"]),
            (self.llama_meta["get_max_seq_len"], self.llama_meta["get_head_dim"]),
        }
        for n in gm.graph.nodes:
            if (
                n.op == "placeholder"
                and len(users := list(n.users)) == 1
                and users[0].meta["val"].size()[-2:] in input_cache_shape
            ):
                n.meta[QCOM_QUANTIZED_IO] = kv_type
            elif n.op == "output":
                for a in n.args[0]:
                    # single head, kv mode
                    if (
                        a.meta["val"].flatten().size()[0]
                        == self.llama_meta["get_head_dim"]
                    ):
                        a.meta[QCOM_QUANTIZED_IO] = kv_type
                    # single head, batch_prefill mode
                    elif a.meta["val"].flatten().size()[0] == self.llama_meta[
                        "get_head_dim"
                    ] * (self.llama_meta["get_max_seq_len"] - 1):
                        a.meta[QCOM_QUANTIZED_IO] = kv_type

    def quantize(self, quant_dtype, custom_annotations=()):
        self.quant_dtype = quant_dtype
        quantizer = make_quantizer(
            quant_dtype=quant_dtype,
            per_channel_conv=True,
            per_channel_linear=True,
            act_observer=MinMaxObserver,
        )
        quantizer.add_custom_quant_annotations(custom_annotations)

        self.has_quant_io = True
        fx_graph_module = None

        with torch.no_grad():
            fx_graph_module = torch.export.export(
                self.llama_model, self.inputs
            ).module()
            fx_graph_module = prepare_pt2e(fx_graph_module, quantizer)
        print("Quantizing the model...")

        calibrate(
            self.get_example_inputs(self.llama_meta["get_use_kv_cache"]),
            args.prompt,
            fx_graph_module,
            tokenizer_model_path=args.tokenizer_model,
            max_seq_len=args.seq_len,
        )

        self.llama_model = convert_pt2e(fx_graph_module)

    def lowering_modules(
        self, work_space, kv_type=torch.uint8, soc_model=QcomChipset.SM8650
    ):
        executorch_config = ExecutorchBackendConfig(
            passes=[
                BuildQuantIo(),
            ],
            # For shared buffer, user must pass the memory address
            # which is allocated by RPC memory to executor runner.
            # Therefore, won't want to pre-allocate
            # by memory manager in runtime.
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                alloc_graph_output=False,
            ),
            extract_delegate_segments=True,
        )
        with torch.no_grad():
            # backend option
            backend_options = generate_htp_compiler_spec(use_fp16=False)
            compiler_specs = generate_qnn_executorch_compiler_spec(
                soc_model=soc_model,
                backend_options=backend_options,
                shared_buffer=True,
            )
            partitioner = QnnPartitioner(compiler_specs)
            edge_prog = capture_program(self.llama_model, self.inputs)
            self._tag_kv_ios(edge_prog.exported_program.graph_module, kv_type=kv_type)
            edge_prog_mgr = EdgeProgramManager(
                edge_programs={"forward": edge_prog.exported_program},
                constant_methods=self.llama_meta,
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            )
            edge_prog_mgr = edge_prog_mgr.to_backend(partitioner)
            exec_prog_mgr = edge_prog_mgr.to_executorch(config=executorch_config)
            with open(f"{work_space}/{pte_filename}.pte", "wb") as file:
                exec_prog_mgr.write_to_file(file)

    def get_example_inputs(self, use_kv_cache=True):
        return self.llama_model.get_example_inputs(use_kv_cache)


def compile(args):
    os.makedirs(args.artifact, exist_ok=True)
    start_ts = time.time()

    if args.model_mode == "kv":
        use_kv_cache = output_new_cache_only = True
    elif args.model_mode == "batch_prefill" or args.model_mode == "hybrid":
        raise NotImplementedError(
            f"model_mode {args.model_mode} is not implemented yet."
        )
    else:
        raise RuntimeError(f"No such model_mode {args.model_mode}.")

    with open(args.params) as f:
        config = ModelArgs(**json.load(f))
        # TODO: support batch inputs if necessary
        config.max_batch_size = 1
        config.max_seq_len = args.seq_len
        config.use_kv_cache = use_kv_cache
    state_dict = torch.load(
        args.checkpoint, weights_only=True, map_location="cpu", mmap=True
    )
    end_load_ts = time.time()
    print("torch.load checkpoint", end_load_ts - start_ts)

    llama_instance = None
    with torch.device("meta"):
        llama_instance = LlamaModel(config, output_new_cache_only=output_new_cache_only)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    llama_instance.load_state_dict(
        state_dict,
        strict=False,
        assign=True,
    )
    end_load_state_dict_ts = time.time()
    print("instance.load_state_dict", end_load_state_dict_ts - end_load_ts)

    for layer in llama_instance.layers:
        if getattr(layer.attention, "prepare_sha", None):
            layer.attention.prepare_sha()

    kv_type = torch.uint8
    assert args.ptq in [
        "8a8w",
        "16a4w",
    ], f"No support for quant type {args.ptq}. Support 8a8w and 16a4w."
    quant_dtype = getattr(QuantDtype, f"use_{args.ptq}")
    assert args.tokenizer_model is not None, "Need tokenizer model for calibration"

    if args.dtype_override is not None:
        dtype_override = DType[args.dtype_override]
        llama_instance = llama_instance.to(dtype_override.to_torch_dtype())

    llama_instance = convert_linear_to_conv2d(llama_instance)
    single_llama = SingleLlama(llama_instance.eval())

    start_quantize_ts = time.time()
    single_llama.quantize(
        quant_dtype,
        custom_annotations=(
            annotate_matmul_16a8w,
            annotate_linear_16a8w_in_affine_layer,
        ),
    )
    end_quantize_ts = time.time()
    print("single_llama.quantize(quant_dtype)", end_quantize_ts - start_quantize_ts)
    single_llama.lowering_modules(
        args.artifact, kv_type=kv_type, soc_model=get_soc_to_chipset_map()[args.model]
    )
    end_lowering_ts = time.time()
    print("Complete Compile", end_lowering_ts - end_quantize_ts)


def inference(args, pre_gen_pte=""):
    workspace = f"/data/local/tmp/{getpass.getuser()}/executorch/single_llama"

    if args.model_mode != "kv":
        raise NotImplementedError(
            f"model_mode {args.model_mode} is not implemented yet."
        )

    assert args.tokenizer_bin is not None, "Need tokenizer model for interence"
    runner_args = " ".join(
        [
            f"--model_path {pte_filename}.pte",
            "--output_folder_path outputs",
            f"--tokenizer_path {os.path.basename(args.tokenizer_bin)}",
            f'--prompt "{args.prompt}"',
            f"--seq_len {args.seq_len}",
            f"--temperature {args.temperature}",
        ]
    )
    runner_cmd = " ".join(
        [
            f"cd {workspace} &&",
            f"./qnn_llama_runner {runner_args}",
        ]
    )

    pte_path = (
        f"{pre_gen_pte}/{pte_filename}.pte"
        if pre_gen_pte
        else f"{args.artifact}/{pte_filename}.pte"
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
        runner="examples/qualcomm/oss_scripts/llama2/qnn_llama_runner",
    )
    # No pregen inputs, input_list is not required
    adb.push(inputs=[], input_list="", files=[args.tokenizer_bin])
    adb.execute(custom_runner_cmd=runner_cmd)

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)
    outputs = []

    def post_process():
        for f in sorted(
            os.listdir(output_data_folder), key=lambda f: int(f.split("_")[1])
        ):
            with codecs.open(
                os.path.join(output_data_folder, f),
                "r",
                encoding="utf-8",
                errors="replace",
            ) as fdata:
                outputs.append(fdata.read())

    adb.pull(output_path=args.artifact, callback=post_process)

    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(
                json.dumps(
                    {
                        "result": outputs,
                    }
                )
            )
    else:
        for idx, output in enumerate(outputs):
            print(f"Results[{idx}]:\n{output}")


def main():
    parser = setup_common_args_and_variables()
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts and output by this example. Default ./llama2_qnn",
        default="./llama2_qnn",
        type=str,
    )

    parser.add_argument(
        "-P",
        "--ptq",
        help="If specified, will do PTQ quantization. default is 16bits activation and 4bits weight. Support 8a8w and 16a4w.",
        default="16a4w",
    )

    parser.add_argument(
        "--checkpoint",
        help="Pass llama2 checkpoint.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--params",
        help="Pass llama2 params json file.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--tokenizer_bin",
        help="Pass llama2 tokenizer binary.",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--tokenizer_model",
        help="Pass llama2 tokenizer model.",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--prompt",
        help="User prompts for llama2.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--seq_len",
        help="Ouput sequence length for llama2.",
        default=128,
        type=int,
    )

    parser.add_argument(
        "--temperature",
        help="Sampling temperature for llama2.",
        default=0.8,
        type=float,
    )

    parser.add_argument(
        "-d",
        "--dtype-override",
        default="fp32",
        type=str,
        choices=["fp32", "fp16"],
        help="Override the dtype of the model (default is the checkpoint dtype). Options: fp32",
    )

    parser.add_argument(
        "--pre_gen_pte",
        help="Run the Pre-generated llama2 in the given directory",
        type=str,
    )

    parser.add_argument(
        "--num_sharding",
        type=int,
        default=0,
        help="Specify the number of splits by inserting the fallback custom op. The graph will be split evenly by layers.",
    )

    parser.add_argument(
        "--model_mode",
        help="Export and inference batch_prefill mode, kv mode or hybrid(TBD) mode",
        default="kv",
        choices=["batch_prefill", "kv", "hybrid"],
        type=str,
    )

    args = parser.parse_args()
    if args.compile_only and args.pre_gen_pte:
        exit("Cannot set both compile_only and pre_gen_pte as true")

    if args.pre_gen_pte:
        inference(args, args.pre_gen_pte)
        exit(f"Finish the running pre_gen_pte from {args.pre_gen_pte}")

    if args.compile_only:
        compile(args)
        exit(f"Finish compile_only and save to {args.artifact}")

    try:
        compile(args)
        inference(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)


# flake8: noqa: C901
if __name__ == "__main__":
    main()
