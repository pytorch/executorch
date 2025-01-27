# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: reenable pyre after fixing the issues
# pyre-ignore-all-errors

import copy
import getpass
import json
import logging
import os
import sys
import time
from functools import partial
from multiprocessing.connection import Client

import torch

from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner

from executorch.backends.qualcomm.quantizer.custom_annotation import (
    annotate_matmul_16a8w,
)

from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils.constants import QCOM_QUANTIZED_IO
from executorch.backends.qualcomm.utils.utils import (
    capture_program,
    convert_linear_to_conv2d,
    generate_composite_llama_program,
    generate_htp_compiler_spec,
    generate_multi_graph_program,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
    update_spill_fill_size,
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
from executorch.exir.backend.backend_api import to_backend
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


def _kv_calibrate(
    example_inputs,
    user_prompts,
    module: torch.fx.GraphModule,
    tokenizer_model_path="tokenizer.model",
    max_seq_len=512,
):
    sp_model = get_tokenizer(tokenizer_model_path)
    _, atten_mask, _, k_caches, v_caches = example_inputs

    # TODO: change criteria & support batch inputs if necessary
    pos = torch.tensor(0, dtype=torch.int32)
    max_cache_len = max_seq_len - 1
    token_list = sp_model.encode(
        user_prompts, bos=True, eos=False, allowed_special="all"
    )

    with torch.no_grad():
        while token_list[-1] != sp_model.eos_id and pos < max_cache_len:
            logits, new_k_caches, new_v_caches = module(
                torch.full((1, 1), token_list[pos], dtype=torch.int32),
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
                token_list.append(torch.argmax(logits[:, -1], dim=-1).item())

    print(f"calibration data:\n{sp_model.decode(token_list)}")


def _prefill_calibrate(
    example_inputs,
    user_prompts,
    module: torch.fx.GraphModule,
    tokenizer_model_path="tokenizer.model",
    max_seq_len=512,
):
    sp_model = get_tokenizer(tokenizer_model_path)
    _, atten_mask = example_inputs
    max_cache_len = max_seq_len - 1

    # TODO: change criteria & support batch inputs if necessary
    token_list = sp_model.encode(
        user_prompts, bos=True, eos=False, allowed_special="all"
    )
    pos = len(token_list)

    with torch.no_grad():
        while token_list[-1] != sp_model.eos_id and pos < max_cache_len:
            tmp_token_list = torch.tensor(token_list).reshape(1, -1)
            if pos < max_cache_len:
                tmp_token_list = torch.cat(
                    [
                        tmp_token_list,
                        torch.zeros((1, max_cache_len - pos), dtype=torch.int32),
                    ],
                    dim=1,
                )
            logits, new_k_caches, new_v_caches = module(
                tmp_token_list,
                atten_mask,
            )
            token_list.append(torch.argmax(logits[:, pos - 1], dim=-1).item())
            pos += 1

    print(f"calibration data:\n{sp_model.decode(token_list)}")


def calibrate(
    example_inputs,
    user_prompts,
    module: torch.fx.GraphModule,
    tokenizer_model_path="tokenizer.model",
    max_seq_len=512,
):
    if len(example_inputs) == 2:
        _prefill_calibrate(
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
    def __init__(self, llama_model, pte_filename) -> None:
        super().__init__()
        self.llama_model = llama_model
        self.quant_dtype = None
        self.llama_meta = self.llama_model.get_metadata()
        self.has_quant_io = False
        self.pte_filename = pte_filename
        if self.llama_meta["get_use_kv_cache"]:
            tokens, atten_mask, pos_ids, k_caches, v_caches = self.get_example_inputs(
                use_kv_cache=True
            )
            self.inputs = (tokens, atten_mask, pos_ids, *k_caches, *v_caches)
        else:
            tokens, atten_mask = self.get_example_inputs(use_kv_cache=False)
            self.inputs = (tokens, atten_mask)

    def _tag_ios(self, gm: torch.fx.GraphModule, fixed_point_type):
        if not self.has_quant_io:
            return

        # shape of k caches and v caches
        kv_cache_shape = {
            # single head, kv mode input
            (self.llama_meta["get_head_dim"], self.llama_meta["get_max_seq_len"]),
            (self.llama_meta["get_max_seq_len"], self.llama_meta["get_head_dim"]),
            # single head, kv mode output
            (self.llama_meta["get_head_dim"], 1),
            (1, self.llama_meta["get_head_dim"]),
            # single head, bert mode
            (self.llama_meta["get_head_dim"], self.llama_meta["get_max_seq_len"] - 1),
            (self.llama_meta["get_max_seq_len"] - 1, self.llama_meta["get_head_dim"]),
        }
        io_shape = {
            # kv mode
            (
                self.llama_meta["get_max_batch_size"],
                1,
                self.llama_meta["get_vocab_size"],
            ),
            # bert mode
            (
                self.llama_meta["get_max_batch_size"],
                self.llama_meta["get_max_seq_len"] - 1,
                self.llama_meta["get_vocab_size"],
            ),
        }

        atten_mask_shape = {
            # kv mode
            (self.llama_meta["get_max_batch_size"], self.llama_meta["get_max_seq_len"]),
            # bert mode
            (
                self.llama_meta["get_max_seq_len"] - 1,
                self.llama_meta["get_max_seq_len"] - 1,
            ),
        }

        freq_shape = {
            # kv mode
            (1, self.llama_meta["get_head_dim"] // 2),
            # bert mode
            (
                self.llama_meta["get_max_seq_len"] - 1,
                self.llama_meta["get_head_dim"] // 2,
            ),
        }

        freq_op = {
            # kv mode
            exir_ops.edge.aten.select.int,
            # bert mode
            exir_ops.edge.aten.slice_copy.Tensor,
        }

        for n in gm.graph.nodes:
            if n.op == "placeholder":
                if (
                    len(users := list(n.users)) == 1
                    and users[0].meta["val"].size()[-2:] in kv_cache_shape
                ):
                    n.meta[QCOM_QUANTIZED_IO] = fixed_point_type["kv_type"]
                elif n.meta["val"].size() in io_shape:
                    n.meta[QCOM_QUANTIZED_IO] = fixed_point_type["io_type"]
                elif n.meta["val"].size() in atten_mask_shape:
                    n.meta[QCOM_QUANTIZED_IO] = fixed_point_type["io_type"]
            elif n.op == "output":
                for a in n.args[0]:
                    if a.meta["val"].size()[-2:] in kv_cache_shape:
                        a.meta[QCOM_QUANTIZED_IO] = fixed_point_type["kv_type"]
                    elif a.meta["val"].size() in io_shape:
                        a.meta[QCOM_QUANTIZED_IO] = fixed_point_type["io_type"]
                        quant_attrs = a.meta["quant_attrs"]

            # Tag sharding io
            if exir_ops.edge.llama.fallback.default in [
                u.target for u in list(n.users.keys())
            ] + [n.target]:
                n.meta[QCOM_QUANTIZED_IO] = fixed_point_type["io_type"]

            # Tag select op as quantized tensors for freq_sin and freq_cos. It is caused by sharding
            if n.target in freq_op and n.meta["val"].size() in freq_shape:
                n.meta[QCOM_QUANTIZED_IO] = fixed_point_type["io_type"]

        return quant_attrs

    def quantize(self, quant_dtype, args, custom_annotations=()):
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
                self.llama_model, self.inputs, strict=True
            ).module()
            fx_graph_module = prepare_pt2e(fx_graph_module, quantizer)
        logging.info("Quantizing the model...")
        calibrate(
            self.get_example_inputs(self.llama_meta["get_use_kv_cache"]),
            args.prompt,
            fx_graph_module,
            tokenizer_model_path=args.tokenizer_model,
            max_seq_len=self.llama_meta["get_max_seq_len"],
        )

        self.llama_model = convert_pt2e(fx_graph_module)

    def lowering_modules(
        self,
        work_space,
        fixed_point_type,
        use_fp16=False,
        soc_model=QcomChipset.SM8650,
        num_sharding=0,
    ):
        executorch_config = ExecutorchBackendConfig(
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
            backend_options = generate_htp_compiler_spec(
                use_fp16=use_fp16, use_multi_contexts=num_sharding > 0
            )
            compiler_specs = generate_qnn_executorch_compiler_spec(
                soc_model=soc_model,
                backend_options=backend_options,
                shared_buffer=False,
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

            self.quant_attrs = self._tag_ios(
                edge_prog.exported_program.graph_module,
                fixed_point_type=fixed_point_type,
            )
            edge_prog_mgr = EdgeProgramManager(
                edge_programs={"forward": edge_prog.exported_program},
                constant_methods=self.llama_meta,
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            )
            edge_prog_mgr = edge_prog_mgr.to_backend(partitioner)
            if num_sharding > 0:
                update_spill_fill_size(edge_prog_mgr.exported_program())
            exec_prog_mgr = edge_prog_mgr.to_executorch(config=executorch_config)
            with open(f"{work_space}/{pte_filename}.pte", "wb") as file:
                exec_prog_mgr.write_to_file(file)

    def get_example_inputs(self, use_kv_cache=True):
        return self.llama_model.get_example_inputs(use_kv_cache)

    def get_quant_attrs(self):
        return self.quant_attrs


def compile(args, pte_filename):
    os.makedirs(args.artifact, exist_ok=True)
    start_ts = time.time()

    with open(args.params) as f:
        kv_config = ModelArgs(**json.load(f))
        # TODO: support batch inputs if necessary
        kv_config.max_batch_size = 1
        kv_config.max_seq_len = args.kv_seq_len
        kv_config.use_kv_cache = True

        prefill_config = copy.copy(kv_config)
        prefill_config.max_seq_len = args.prefill_seq_len
        prefill_config.use_kv_cache = False

    state_dict = torch.load(
        args.checkpoint, weights_only=True, map_location="cpu", mmap=True
    )

    llama_instance_list = []
    with torch.device("meta"):
        if args.model_mode == "kv":
            llama_instance_list.append(
                LlamaModel(kv_config, output_new_cache_only=True)
            )
        elif args.model_mode == "prefill":
            llama_instance_list.append(
                LlamaModel(prefill_config, output_new_cache_only=False)
            )
        elif args.model_mode == "hybrid":
            llama_instance_list.append(
                LlamaModel(prefill_config, output_new_cache_only=False)
            )
            llama_instance_list.append(
                LlamaModel(kv_config, output_new_cache_only=True)
            )
        else:
            raise RuntimeError(f"No such model_mode {args.model_mode}.")

    if "model" in state_dict:
        state_dict = state_dict["model"]

    for llama_instance in llama_instance_list:
        llama_instance.load_state_dict(
            state_dict,
            strict=False,
            assign=True,
        )
    end_load_ts = time.time()
    logging.info(f"Time for loading checkpoint: {end_load_ts - start_ts}")

    for llama_instance in llama_instance_list:
        for layer in llama_instance.layers:
            if getattr(layer.attention, "prepare_sha", None):
                layer.attention.prepare_sha()
            if getattr(layer.feed_forward, "prepare_feedfoward_conv", None):
                layer.feed_forward.prepare_feedfoward_conv()

    use_fp16 = True
    fixed_point_type = {"kv_type": torch.float32, "io_type": torch.float32}
    if args.ptq:
        use_fp16 = False
        fixed_point_type["kv_type"] = torch.uint8
        if args.ptq == "8a8w":
            fixed_point_type["io_type"] = torch.uint8
        elif args.ptq == "16a4w":
            fixed_point_type["io_type"] = torch.uint16
        else:
            assert args.ptq in [
                "8a8w",
                "16a4w",
            ], f"No support for quant type {args.ptq}. Support 8a8w and 16a4w."
        quant_dtype = getattr(QuantDtype, f"use_{args.ptq}")

    assert args.tokenizer_model is not None, "Need tokenizer model for calibration"

    if args.dtype_override is not None:
        dtype_override = DType[args.dtype_override]
        for i in range(len(llama_instance_list)):
            llama_instance_list[i] = llama_instance_list[i].to(
                dtype_override.to_torch_dtype()
            )

    for i in range(len(llama_instance_list)):
        llama_instance_list[i] = convert_linear_to_conv2d(llama_instance_list[i])
        llama_instance_list[i] = SingleLlama(
            llama_instance_list[i].eval(), pte_filename
        )

    if args.ptq:
        start_quantize_ts = time.time()
        for llama_instance in llama_instance_list:
            llama_instance.quantize(
                quant_dtype=quant_dtype,
                args=args,
                custom_annotations=(
                    partial(
                        annotate_matmul_16a8w,
                        traverse_input1=llama_instance.llama_meta["get_use_kv_cache"],
                    ),
                ),
            )
        end_quantize_ts = time.time()
        logging.info(f"Time for quantizing: {end_quantize_ts - start_quantize_ts}")

    start_lowering_ts = time.time()
    quant_attrs = None

    if len(llama_instance_list) == 1:
        llama_instance_list[0].lowering_modules(
            args.artifact,
            fixed_point_type,
            use_fp16=use_fp16,
            soc_model=get_soc_to_chipset_map()[args.model],
            num_sharding=args.num_sharding,
        )
        quant_attrs = llama_instance_list[0].get_quant_attrs()
    else:
        sample_inputs_list = [
            llama_instace.inputs for llama_instace in llama_instance_list
        ]
        edge_progs = [
            capture_program(llama_instance.llama_model, sample_input)
            for llama_instance, sample_input in zip(
                llama_instance_list, sample_inputs_list
            )
        ]

        if args.num_sharding > 0:
            for i in range(len(llama_instance_list)):
                model_sharding.split_graph(
                    edge_progs[i].exported_program,
                    llama_instance_list[i].llama_meta["get_n_layers"],
                    shares=args.num_sharding,
                )

        for i in range(len(llama_instance_list)):
            quant_attrs = llama_instance_list[i]._tag_ios(
                edge_progs[i].exported_program.graph_module,
                fixed_point_type,
            )
        backend_options = generate_htp_compiler_spec(
            use_fp16=use_fp16, use_multi_contexts=args.num_sharding > 0
        )
        graph_names = ["prefill_forward", "kv_forward"]
        compiler_specs = [
            generate_qnn_executorch_compiler_spec(
                soc_model=get_soc_to_chipset_map()[args.model],
                backend_options=backend_options,
                shared_buffer=True,
                multiple_graphs=True,
                graph_name=graph_name,
            )
            for graph_name in graph_names
        ]
        skip_node_op_set = {"llama.fallback.default"}
        exported_programs = [
            to_backend(
                edge_prog.exported_program,
                QnnPartitioner(compiler_specs[i], skip_node_op_set=skip_node_op_set),
            )
            for i, edge_prog in enumerate(edge_progs)
        ]
        if args.num_sharding > 0:
            for exported_program in exported_programs:
                update_spill_fill_size(exported_program)

        executorch_config = ExecutorchBackendConfig(
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

        lower_module_dict = {name: [] for name in graph_names}
        call_delegate_inputs_dict = {name: [] for name in graph_names}
        call_delegate_node_name_dict = {name: [] for name in graph_names}
        outputs_dict = {name: [] for name in graph_names}
        input_nodes_dict = {name: [] for name in graph_names}
        for prog, graph_name in zip(exported_programs, graph_names):
            for node in prog.graph_module.graph.nodes:
                if (
                    node.op == "call_function"
                    and "executorch_call_delegate" in node.name
                ):
                    call_delegate_node_name_dict[graph_name].append(node.name)
                    call_delegate_inputs_list = []
                    for arg in node.args:
                        if arg.op == "call_function":
                            while "getitem" not in arg.name:
                                arg = arg.args[0]
                            call_delegate_inputs_list.append(
                                (arg.args[0].name, arg.args[1])
                            )
                        elif arg.op == "placeholder":
                            call_delegate_inputs_list.append((arg.name, None))
                        # No extra needs to do for get_attr node
                    call_delegate_inputs_dict[graph_name].append(
                        call_delegate_inputs_list
                    )
                elif node.op == "output":
                    for arg in node.args[0]:
                        outputs_dict[graph_name].append((arg.args[0].name, arg.args[1]))

        if args.num_sharding > 0:
            bundle_progs_list = []
            for num in range(args.num_sharding - 1, -1, -1):
                processed_bytes = []
                for prog, graph_name in zip(exported_programs, graph_names):
                    processed_bytes.append(
                        getattr(
                            prog.graph_module, f"lowered_module_{num}"
                        ).processed_bytes
                    )

                    call_delegate_node = [
                        list(node.users.keys())[0]
                        for node in prog.graph_module.graph.nodes
                        if node.op == "get_attr"
                        and node.name == f"lowered_module_{num}"
                    ]
                    input_nodes_dict[graph_name] = [
                        node
                        for node in call_delegate_node[0].args
                        if node.op == "placeholder"
                    ]

                prog_mgr, bundle_progs = generate_multi_graph_program(
                    compiler_specs=compiler_specs[0],
                    processed_bytes=processed_bytes,
                    input_nodes_dict=input_nodes_dict,
                    backend_config=executorch_config,
                    constant_methods=llama_instance_list[
                        1
                    ].llama_meta,  # kv method meta
                )
                bundle_progs_list.append(bundle_progs)
                for graph_name in graph_names:
                    lower_module_dict[graph_name].append(
                        prog_mgr.exported_program(graph_name).graph_module._modules.get(
                            "lowered_module_0"
                        )
                    )

            exec_prog = generate_composite_llama_program(
                graph_names=graph_names,
                sample_inputs_list=sample_inputs_list,
                lower_module_dict=lower_module_dict,
                call_delegate_node_name_dict=call_delegate_node_name_dict,
                call_delegate_inputs_dict=call_delegate_inputs_dict,
                outputs_dict=outputs_dict,
                backend_config=executorch_config,
                constant_methods=llama_instance_list[1].llama_meta,  # kv method meta
            )
            with open(f"{args.artifact}/{pte_filename}.pte", "wb") as file:
                exec_prog.write_to_file(file)
        else:
            processed_bytes = []
            input_nodes_dict = {name: [] for name in graph_names}
            output_nodes_dict = {name: [] for name in graph_names}
            for prog, graph_name in zip(exported_programs, graph_names):
                processed_bytes.append(
                    prog.graph_module.lowered_module_0.processed_bytes
                )
                input_nodes_dict[graph_name] = [
                    node
                    for node in prog.graph_module.graph.nodes
                    if node.op == "placeholder"
                ]
                output_nodes_dict[graph_name] = [
                    node
                    for node in prog.graph_module.graph.nodes
                    if node.op == "output"
                ]

            prog_mgr, _ = generate_multi_graph_program(
                compiler_specs=compiler_specs[0],
                processed_bytes=processed_bytes,
                input_nodes_dict=input_nodes_dict,
                output_nodes_dict=output_nodes_dict,
                backend_config=executorch_config,
                constant_methods=llama_instance_list[1].llama_meta,  # kv method meta
            )
            with open(f"{args.artifact}/{pte_filename}.pte", "wb") as file:
                prog_mgr.write_to_file(file)

    end_lowering_ts = time.time()
    logging.info(f"Time for compiling: {end_lowering_ts - start_lowering_ts}")
    return quant_attrs


def inference(args, quant_attrs, pte_filename, pre_gen_pte=""):
    workspace = f"/data/local/tmp/{getpass.getuser()}/executorch/single_llama"

    if args.model_mode == "prefill":
        eval_mode = 0
    elif args.model_mode == "kv":
        eval_mode = 1
    elif args.model_mode == "hybrid":
        eval_mode = 2
    else:
        raise RuntimeError(f"No such model_mode {args.model_mode}.")

    seq_len = args.prefill_seq_len if args.model_mode == "prefill" else args.kv_seq_len
    runner_args = " ".join(
        [
            f"--model_path {pte_filename}.pte",
            "--output_path outputs/outputs.txt",
            f"--tokenizer_path {os.path.basename(args.tokenizer_model)}",
            f'--prompt "{args.prompt}"',
            f"--seq_len {seq_len}",
            f"--eval_mode {eval_mode}",
            f"--temperature {args.temperature}",
            f"--system_prompt '{args.system_prompt}'",
            f"--logits_scale {quant_attrs['scale']}",
            f"--logits_offset {quant_attrs['zero_point']}",
        ]
    )
    runner_cmd = " ".join(
        [
            f"cd {workspace} &&",
            f"./qnn_llama3_2_runner {runner_args}",
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
        runner=f"examples/qualcomm/oss_scripts/llama3_2/qnn_llama3_2_runner",
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


def main():
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
        "--system_prompt",
        help="Tells the model what kind of assistant it should be. For example, You are a helpful AI assistant for travel tips and recommendations. Default is None",
        default="",
        type=str,
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

    parser.add_argument(
        "--model_mode",
        help="Export and inference prefill mode, kv mode or hybrid mode",
        default="kv",
        choices=["prefill", "kv", "hybrid"],
        type=str,
    )

    parser.add_argument(
        "--prefill_seq_len",
        help="Ouput sequence length for llama. Use this option for prefill or hybrid mode",
        default=32,
        type=int,
    )

    parser.add_argument(
        "--kv_seq_len",
        help="Ouput sequence length for llama. Use this option for kv or hybrid mode",
        default=512,
        type=int,
    )

    args = parser.parse_args()
    if args.compile_only and args.pre_gen_pte:
        exit("Cannot set both compile_only and pre_gen_pte as true")

    if args.model_mode == "kv":
        pte_filename = "kv_llama3_2_qnn"
    elif args.model_mode == "prefill":
        pte_filename = "prefill_llama3_2_qnn"
    elif args.model_mode == "hybrid":
        assert (
            args.kv_seq_len >= args.prefill_seq_len
        ), "Please ensure kv_seq_len is >= prefill_seq_len"
        pte_filename = "hybrid_llama3_2_qnn"
    else:
        raise RuntimeError(f"No such model_mode {args.model_mode}.")

    if args.pre_gen_pte:
        quant_attrs = json.load(
            open(f"{args.pre_gen_pte}/{pte_filename}_quant_attrs.txt")
        )
        inference(args, quant_attrs, pte_filename, args.pre_gen_pte)
        exit(f"Finish the running pre_gen_pte from {args.pre_gen_pte}")

    if args.compile_only:
        quant_attrs = compile(args, pte_filename)
        if quant_attrs:
            json.dump(
                {
                    "scale": quant_attrs["scale"],
                    "zero_point": quant_attrs["zero_point"],
                },
                open(f"{args.artifact}/{pte_filename}_quant_attrs.txt", "w"),
            )
        else:
            logging.warning("Quant attributes of the logit is None.")
        exit(f"Finish compile_only and save to {args.artifact}")

    try:
        quant_attrs = compile(args, pte_filename)
        if quant_attrs:
            logging.info(
                f"Logit scale: {quant_attrs['scale']}; Logit offset: {quant_attrs['zero_point']}"
            )
            json.dump(
                {
                    "scale": quant_attrs["scale"],
                    "zero_point": quant_attrs["zero_point"],
                },
                open(f"{args.artifact}/{pte_filename}_quant_attrs.txt", "w"),
            )
        else:
            logging.warning("Quant attributes of the logit is None.")
        inference(args, quant_attrs, pte_filename)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)


# flake8: noqa: C901
if __name__ == "__main__":
    main()
