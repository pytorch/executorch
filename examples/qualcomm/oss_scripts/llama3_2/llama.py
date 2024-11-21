# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import getpass
import json
import logging
import os

import sys
import time
from multiprocessing.connection import Client

import torch
from executorch.backends.qualcomm._passes.build_quant_io import BuildQuantIo

from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner

from executorch.backends.qualcomm.quantizer.custom_annotation import (
    annotate_matmul_16a8w,
    custom_annotate_llama_last_conv_16a8w,
)

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
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.extension.llm.custom_ops import model_sharding
from executorch.extension.llm.export.builder import DType
from executorch.extension.llm.tokenizer.utils import get_tokenizer

from torch.ao.quantization.observer import MinMaxObserver
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

sys.setrecursionlimit(4096)
FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

pte_filename = "llama3_2_qnn"


def calibrate(
    example_inputs,
    user_prompts,
    module: torch.fx.GraphModule,
    tokenizer_model_path="tokenizer.model",
):
    sp_model = get_tokenizer(tokenizer_model_path)
    _, _, atten_mask, k_caches, v_caches = example_inputs

    # TODO: change criteria & support batch inputs if necessary
    pos = torch.tensor(0, dtype=torch.int32)
    token_list = sp_model.encode(user_prompts, bos=True, eos=False)

    with torch.no_grad():
        while token_list[-1] != sp_model.eos_id and pos < 511:
            logits, new_k_caches, new_v_caches = module(
                torch.full((1, 1), token_list[pos], dtype=torch.int32),
                torch.full((1, 1), pos),
                atten_mask,
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
                token_list.append(torch.argmax(logits[:, -1], dim=-1).item())

    print(f"calibration data:\n{sp_model.decode(token_list)}")


class SingleLlama:
    def __init__(self, llama_model) -> None:
        super().__init__()
        self.llama_model = llama_model
        self.quant_dtype = None
        self.llama_meta = self.llama_model.get_metadata()
        self.has_quant_io = False
        tokens, pos_ids, atten_mask, k_caches, v_caches = self.get_example_inputs()
        self.inputs = (tokens, pos_ids, atten_mask, *k_caches, *v_caches)

    def _tag_kv_ios(self, gm: torch.fx.GraphModule, kv_type, sharding_type):
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
                    if (
                        a.meta["val"].flatten().size()[0]
                        == self.llama_meta["get_head_dim"]
                    ):
                        a.meta[QCOM_QUANTIZED_IO] = kv_type

            # Tag sharding io
            if exir_ops.edge.llama.fallback.default in [
                u.target for u in list(n.users.keys())
            ] + [n.target]:
                n.meta[QCOM_QUANTIZED_IO] = sharding_type

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
        logging.info("Quantizing the model...")
        calibrate(
            self.get_example_inputs(),
            args.prompt,
            fx_graph_module,
            tokenizer_model_path=args.tokenizer_model,
        )

        self.llama_model = convert_pt2e(fx_graph_module)

    def lowering_modules(
        self,
        work_space,
        kv_type=torch.uint8,
        sharding_type=torch.uint16,
        use_fp16=False,
        soc_model=QcomChipset.SM8650,
        num_sharding=0,
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
            backend_options = generate_htp_compiler_spec(use_fp16=use_fp16)
            compiler_specs = generate_qnn_executorch_compiler_spec(
                soc_model=soc_model,
                backend_options=backend_options,
                shared_buffer=True,
            )
            skip_node_op_set = {"llama.fallback.default"}
            partitioner = QnnPartitioner(
                compiler_specs, skip_node_op_set=skip_node_op_set
            )
            edge_prog = capture_program(
                self.llama_model, self.inputs, custom_pass_config=frozenset()
            )

            if num_sharding > 0:
                model_sharding.split_graph(
                    edge_prog.exported_program,
                    self.llama_meta["get_n_layers"],
                    shares=num_sharding,
                )

            self._tag_kv_ios(
                edge_prog.exported_program.graph_module,
                kv_type=kv_type,
                sharding_type=sharding_type,
            )
            edge_prog_mgr = EdgeProgramManager(
                edge_programs={"forward": edge_prog.exported_program},
                constant_methods=self.llama_meta,
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            )
            edge_prog_mgr = edge_prog_mgr.to_backend(partitioner)
            exec_prog_mgr = edge_prog_mgr.to_executorch(config=executorch_config)
            with open(f"{work_space}/{pte_filename}.pte", "wb") as file:
                exec_prog_mgr.write_to_file(file)

    def get_example_inputs(self):
        return self.llama_model.get_example_inputs()


def compile(args):
    os.makedirs(args.artifact, exist_ok=True)
    start_ts = time.time()
    with open(args.params) as f:
        config = ModelArgs(**json.load(f))
        # TODO: support batch inputs if necessary
        config.max_batch_size = 1
        config.max_seq_len = 512
    state_dict = torch.load(
        args.checkpoint, weights_only=True, map_location="cpu", mmap=True
    )

    llama_instance = None
    with torch.device("meta"):
        llama_instance = LlamaModel(config, output_new_cache_only=True)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    llama_instance.load_state_dict(
        state_dict,
        strict=False,
        assign=True,
    )
    end_load_ts = time.time()
    logging.info(f"Time for loading checkpoint: {end_load_ts - start_ts}")

    for layer in llama_instance.layers:
        if getattr(layer.attention, "prepare_sha", None):
            layer.attention.prepare_sha()

    use_fp16 = False
    if args.ptq != None:
        kv_type = torch.uint8
        if args.ptq == "8a8w":
            sharding_type = torch.uint8
        elif args.ptq == "16a4w":
            sharding_type = torch.uint16
        else:
            assert args.ptq in [
                "8a8w",
                "16a4w",
            ], f"No support for quant type {args.ptq}. Support 8a8w and 16a4w."
        quant_dtype = getattr(QuantDtype, f"use_{args.ptq}")
    else:
        use_fp16 = True
        kv_type = torch.float32
        sharding_type = torch.float32
    assert args.tokenizer_model is not None, "Need tokenizer model for calibration"

    if args.dtype_override is not None:
        dtype_override = DType[args.dtype_override]
        llama_instance = llama_instance.to(dtype_override.to_torch_dtype())

    llama_instance = convert_linear_to_conv2d(llama_instance)
    single_llama = SingleLlama(llama_instance.eval())

    if args.ptq != None:
        start_quantize_ts = time.time()
        single_llama.quantize(
            quant_dtype,
            custom_annotations=(
                custom_annotate_llama_last_conv_16a8w,
                annotate_matmul_16a8w,
            ),
        )
        end_quantize_ts = time.time()
        logging.info(f"Time for quantizing: {end_quantize_ts - start_quantize_ts}")

    start_lowering_ts = time.time()
    single_llama.lowering_modules(
        args.artifact,
        kv_type=kv_type,
        sharding_type=sharding_type,
        use_fp16=use_fp16,
        soc_model=get_soc_to_chipset_map()[args.model],
        num_sharding=args.num_sharding,
    )
    end_lowering_ts = time.time()
    logging.info(f"Time for compiling: {end_lowering_ts - start_lowering_ts}")


def inference(args, pre_gen_pte=""):
    workspace = f"/data/local/tmp/{getpass.getuser()}/executorch/single_llama"

    runner_args = " ".join(
        [
            f"--model_path {pte_filename}.pte",
            "--output_path outputs/outputs.txt",
            f"--tokenizer_path {os.path.basename(args.tokenizer_model)}",
            f'--prompt "{args.prompt}"',
            f"--seq_len {args.seq_len}",
            f"--temperature {args.temperature}",
        ]
    )
    runner_cmd = " ".join(
        [
            f"cd {workspace} &&",
            f"./qnn_llama3_2_{args.model_size.lower()}_runner {runner_args}",
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
        runner=f"examples/qualcomm/oss_scripts/llama3_2/qnn_llama3_2_{args.model_size.lower()}_runner",
    )
    # No pregen inputs, input_list is not required
    adb.push(inputs=[], input_list="", files=[args.tokenizer_model])
    adb.execute(custom_runner_cmd=runner_cmd)

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)
    outputs = []

    def post_process():
        with open(f"{args.artifact}/outputs/outputs.txt", "r") as f:
            outputs.append(f.read())

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
            logging.info(f"Results[{idx}]:\n{output}")


# flake8: noqa: C901
if __name__ == "__main__":
    parser = setup_common_args_and_variables()
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts and output by this example. Default ./llama3_2_qnn",
        default="./llama3_2_qnn",
        type=str,
    )

    parser.add_argument(
        "-P",
        "--ptq",
        help="If specified, will do PTQ quantization. default is 16bits activation and 4bits weight. Support 8a8w and 16a4w.",
        type=str,
    )

    parser.add_argument(
        "--checkpoint",
        help="Pass llama checkpoint.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--params",
        help="Pass llama params json file.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--model_size",
        help="Determine what runner be used. For llama 3.2, we only support 1B/3B. ",
        choices=["1B", "3B"],
        required=True,
        type=str,
    )

    parser.add_argument(
        "--tokenizer_model",
        help="Pass llama tokenizer model.",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--prompt",
        help="User prompts for llama.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--seq_len",
        help="Ouput sequence length for llama.",
        default=128,
        type=int,
    )

    parser.add_argument(
        "--temperature",
        help="Sampling temperature for llama.",
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
        help="Run the Pre-generated llama in the given directory",
        type=str,
    )

    parser.add_argument(
        "--num_sharding",
        type=int,
        default=0,
        help="Specify the number of splits by inserting the fallback custom op. The graph will be split evenly by layers.",
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
