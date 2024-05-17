# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import codecs
import gc
import getpass
import json
import os
import shutil
import stat
import sys
from pathlib import Path

sys.setrecursionlimit(4096)

import time
from typing import List, Tuple

import torch

from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
from executorch.backends.qualcomm.passes.build_quant_io import BuildQuantIo
from executorch.backends.qualcomm.passes.utils import q_io_key

from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
from executorch.backends.qualcomm.quantizer.utils import get_16a4w_qnn_ptq_config
from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import (
    QcomChipset,
)
from executorch.backends.qualcomm.utils.utils import (
    capture_program,
    convert_linear_to_conv2d,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
)
from executorch.examples.models.llama2.builder import DType
from executorch.examples.models.llama2.llama_transformer import precompute_freqs_cis
from executorch.examples.qualcomm.llama2.model.static_llama import LlamaModel, ModelArgs
from executorch.examples.qualcomm.scripts.utils import (
    make_output_dir,
    setup_common_args_and_variables,
    SimpleADB,
)
from executorch.exir import EdgeCompileConfig, EdgeProgramManager
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass

from sentencepiece import SentencePieceProcessor
from torch.ao.quantization.observer import MinMaxObserver
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e


def annotate_matmul_16a8w(gm: torch.fx.GraphModule) -> None:
    """
    This function is specific for matmul op 16a8w.
    """
    from typing import Sequence

    from executorch.backends.qualcomm.quantizer.quantizer import (
        get_16a8w_qnn_ptq_config,
        get_default_8bit_qnn_ptq_config,
        QuantizationConfig,
    )
    from executorch.backends.qualcomm.quantizer.utils import QUANT_ANNOTATION_KEY
    from torch.ao.quantization.quantizer import (
        QuantizationAnnotation,
        SharedQuantizationSpec,
    )
    from torch.fx import Node

    def annotate_matmul(node: Node, quantization_config: QuantizationConfig):
        input_qspec_map = {}
        input_act = node.args[0]
        assert isinstance(input_act, Node)
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

        assert isinstance(input_nodes, Sequence)

        first_input_node = input_nodes[0]
        input_qspec_map = {}
        assert isinstance(first_input_node, Node)
        assert isinstance(node, Node)
        input_qspec_map[first_input_node] = quantization_config.input_activation
        share_qparams_with_input_act0_qspec = SharedQuantizationSpec(
            (first_input_node, node)
        )

        for input_node in input_nodes[1:]:
            if input_node not in input_qspec_map:
                assert isinstance(input_node, Node)
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
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = quantization_config.input_activation

        node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )

    def annotate_matmul_input1(node: Node):
        quantization_config_8a8w = get_default_8bit_qnn_ptq_config(act_symmetric=True)
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
    from executorch.backends.qualcomm.quantizer.quantizer import (
        get_ptq_per_channel_weight_config,
        QuantizationConfig,
    )
    from executorch.backends.qualcomm.quantizer.utils import QUANT_ANNOTATION_KEY
    from torch.ao.quantization.quantizer import QuantizationAnnotation
    from torch.fx import Node

    def annotate_conv2d(node: Node, quantization_config: QuantizationConfig) -> None:
        input_qspec_map = {}
        input_act = node.args[0]
        assert isinstance(input_act, Node)
        input_spec = quantization_config.input_activation
        input_qspec_map[input_act] = input_spec

        weight = node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = quantization_config.weight

        if len(node.args) > 2:
            bias = node.args[2]
            if isinstance(bias, Node):
                if callable(quantization_config.bias):
                    input_qspec_map[bias] = quantization_config.bias(node)
                else:
                    input_qspec_map[bias] = quantization_config.bias

        node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )

    quantization_config_16a8w_per_channel = get_ptq_per_channel_weight_config(
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


def calibrate(
    example_inputs, n_heads, layers_per_ctx, modules: List[torch.fx.GraphModule]
):
    sp_model = SentencePieceProcessor(model_file="tokenizer.model")
    _, _, freqs_cos, freqs_sin, atten_mask, k_caches, v_caches = example_inputs

    # TODO: change criteria & support batch inputs if necessary
    pos = torch.tensor(0, dtype=torch.int32)
    token_list = [sp_model.bos_id()]
    user_prompts = ["Once"]
    for prompt in user_prompts:
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
        while token_list[-1] != sp_model.eos_id() and pos < 128:
            hidden_states = modules[0](torch.full((1, 1), token_list[pos]))
            input_pos = torch.full((1, 1), pos)
            k_caches_o_list = []
            v_caches_o_list = []
            for i, decode_module in enumerate(modules[1:-1]):
                offset = i * layers_per_ctx * n_heads
                k_caches_i = k_caches[offset : offset + layers_per_ctx * n_heads]
                v_caches_i = v_caches[offset : offset + layers_per_ctx * n_heads]
                hidden_states, k_caches_o, v_caches_o = decode_module(
                    hidden_states,
                    freqs_cos[input_pos][0],
                    freqs_sin[input_pos][0],
                    atten_mask,
                    k_caches_i,
                    v_caches_i,
                )
                k_caches_o_list.extend(k_caches_o)
                v_caches_o_list.extend(v_caches_o)

            logits = modules[-1](hidden_states)
            # k_caches have been transposed ahead, the shpae is [batch, head_dim, seq-1]
            k_caches = [
                torch.cat([k_cache[:, :, 1:], k_caches_o_list[i]], dim=-1)
                for i, k_cache in enumerate(k_caches)
            ]
            v_caches = [
                torch.cat([v_cache[:, 1:, :], v_caches_o_list[i]], dim=1)
                for i, v_cache in enumerate(v_caches)
            ]

            pos += 1
            atten_mask[0][-pos - 1] = 0
            if pos >= len(token_list):
                token_list.append(torch.argmax(logits[:, -1], dim=-1).item())

    print(f"calibration data:\n{sp_model.decode(token_list)}")


class CompositeLlama:
    def __init__(self, division, llama_model) -> None:
        super().__init__()
        self.division = division
        self.layers_per_ctx = llama_model.n_layers // division
        self.llama_model = llama_model
        self.quant_dtype = None
        self.split_modules, self.split_inputs = [], []
        self.llama_meta = self.llama_model.get_metadata()
        self.has_quant_io = False

    def split_llama(self):
        def get_block_module(llama, indexes):
            class LlamaBlock(torch.nn.Module):
                def __init__(self, llama, indexes) -> None:
                    super().__init__()
                    self.llama = llama
                    self.indexes = indexes

                def forward(
                    self,
                    hidden_states: torch.Tensor,
                    freqs_cos: torch.Tensor,
                    freqs_sin: torch.Tensor,
                    atten_mask: torch.Tensor,
                    k_caches: List[torch.Tensor],
                    v_caches: List[torch.Tensor],
                ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
                    output_k_cache, output_v_cache = [], []
                    for i, ind in enumerate(self.indexes):
                        offset = i * self.llama.n_heads
                        k_in = k_caches[offset : offset + self.llama.n_heads]
                        v_in = v_caches[offset : offset + self.llama.n_heads]
                        hidden_states, k, v = self.llama.layers[ind](
                            x=hidden_states,
                            freqs_cos=freqs_cos,
                            freqs_sin=freqs_sin,
                            atten_mask=atten_mask,
                            k_caches=k_in,
                            v_caches=v_in,
                        )
                        output_k_cache.extend(k)
                        output_v_cache.extend(v)

                    return hidden_states, output_k_cache, output_v_cache

            return LlamaBlock(llama, indexes)

        def get_affine_module(llama):
            class LlamaAffine(torch.nn.Module):
                def __init__(self, llama) -> None:
                    super().__init__()
                    self.llama = llama

                def forward(self, hidden_states):
                    hidden_states = self.llama.norm(hidden_states)
                    logits = self.llama.output(hidden_states)
                    return logits

            return LlamaAffine(llama)

        tokens, pos_ids, freqs_cos, freqs_sin, atten_mask, k_caches, v_caches = (
            self.get_example_inputs()
        )

        with torch.no_grad():
            # embedding
            self.split_modules.append(self.llama_model.tok_embeddings)
            self.split_inputs.append((tokens,))

            # attentions
            for i in range(self.division):
                llama_block = get_block_module(
                    self.llama_model,
                    [*range(self.layers_per_ctx * i, self.layers_per_ctx * (i + 1))],
                )
                offset = i * self.layers_per_ctx * self.llama_model.n_heads
                k_caches_in = k_caches[
                    offset : offset + self.layers_per_ctx * self.llama_model.n_heads
                ]
                v_caches_in = v_caches[
                    offset : offset + self.layers_per_ctx * self.llama_model.n_heads
                ]
                self.split_modules.append(llama_block)
                self.split_inputs.append(
                    (
                        self.llama_model.tok_embeddings(tokens),
                        freqs_cos[pos_ids][0],
                        freqs_sin[pos_ids][0],
                        atten_mask,
                        k_caches_in,
                        v_caches_in,
                    )
                )

            # affine layer
            affine_block = get_affine_module(self.llama_model)
            self.split_modules.append(affine_block)
            self.split_inputs.append((self.llama_model.tok_embeddings(tokens),))

    def _tag_kv_ios(self, gm: torch.fx.GraphModule, kv_type=torch.float32):
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
                n.meta[q_io_key] = kv_type
            elif n.op == "output":
                for a in n.args[0]:
                    if (
                        a.meta["val"].flatten().size()[0]
                        == self.llama_meta["get_head_dim"]
                    ):
                        a.meta[q_io_key] = kv_type

    def quantize(self, quant_dtype, custom_annotations=()):
        self.quant_dtype = quant_dtype
        quantizer = QnnQuantizer()
        quantizer.set_per_channel_linear_quant(True)
        quantizer.set_per_channel_conv_quant(True)

        if quant_dtype == QuantDtype.use_8a8w:
            pass  # default setting
        elif quant_dtype == QuantDtype.use_16a4w:
            quantizer.add_16bit_quant_ops(quantizer.SUPPORTED_OPS)
            quantizer.set_bit16_op_quant_config(
                get_16a4w_qnn_ptq_config(act_observer=MinMaxObserver)
            )
            quantizer.set_per_channel_weight_dtype(weight_dtype_for_16bit_act="int4")
        else:
            raise AssertionError(f"No support for QuantDtype {quant_dtype}.")
        quantizer.add_custom_quant_annotations(custom_annotations)

        self.has_quant_io = True
        split_fx_graph_modules = []

        with torch.no_grad():
            for nn_module, capture_inputs in zip(self.split_modules, self.split_inputs):
                fx_graph_module = torch._export.capture_pre_autograd_graph(
                    nn_module, capture_inputs
                )
                fx_graph_module = prepare_pt2e(fx_graph_module, quantizer)
                split_fx_graph_modules.append(fx_graph_module)
        print("Quantizing the model...")
        calibrate(
            self.get_example_inputs(),
            self.llama_model.n_heads,
            self.layers_per_ctx,
            split_fx_graph_modules,
        )

        self.split_modules = [
            convert_pt2e(fx_graph_module) for fx_graph_module in split_fx_graph_modules
        ]
        del self.llama_model

    def lowering_modules(self, work_space, kv_type=torch.float32):

        executorch_config = ExecutorchBackendConfig(
            passes=[
                BuildQuantIo(),
            ],
            extract_constant_segment=False,
            # For shared buffer, user must pass the memory address
            # which is allocated by RPC memory to executor runner.
            # Therefore, won't want to pre-allocate
            # by memory manager in runtime.
            memory_planning_pass=MemoryPlanningPass(
                memory_planning_algo="greedy",
                alloc_graph_input=False,
                alloc_graph_output=False,
            ),
            extract_delegate_segments=True,
        )
        pte_filename_list = []
        index = len(self.split_modules)
        with torch.no_grad():
            while index > 0:
                # backend option
                backend_options = generate_htp_compiler_spec(
                    use_fp16=True if self.quant_dtype is None else False,
                    use_multi_contexts=True,
                )
                compiler_specs = generate_qnn_executorch_compiler_spec(
                    soc_model=QcomChipset.SM8650,
                    backend_options=backend_options,
                    # saver=True if index==5 else False
                )
                partitioner = QnnPartitioner(compiler_specs)
                pte_filename = f"llama2_qnn_{index-1}"
                edge_prog = capture_program(
                    self.split_modules[index - 1], self.split_inputs[index - 1]
                )
                self._tag_kv_ios(
                    edge_prog.exported_program.graph_module, kv_type=kv_type
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

                del edge_prog
                del edge_prog_mgr
                del exec_prog_mgr
                self.split_modules.pop()
                self.split_inputs.pop()
                gc.collect(generation=2)
                pte_filename_list.insert(0, f"{work_space}/{pte_filename}.pte")
                index -= 1
        return pte_filename_list

    def get_example_inputs(self):
        tokens, pos_ids, atten_mask, k_caches, v_caches = (
            self.llama_model.get_example_inputs()
        )
        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.llama_model.dim // self.llama_model.n_heads,
            self.llama_model.max_seq_len,
            self.llama_model.rope_freq_base,
        )
        return (tokens, pos_ids, freqs_cos, freqs_sin, atten_mask, k_caches, v_caches)

    def get_export_inputs(self):
        tokens, pos_ids, atten_mask, k_caches, v_caches = (
            self.llama_model.get_export_inputs()
        )
        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.llama_model.dim // self.llama_model.n_heads,
            self.llama_model.max_seq_len,
            self.llama_model.rope_freq_base,
        )
        export_inputs = [tokens, pos_ids, freqs_cos, freqs_sin, atten_mask]
        for i in range(self.division):
            offset = i * self.layers_per_ctx * self.llama_model.n_heads
            k_caches_in = k_caches[
                offset : offset + self.layers_per_ctx * self.llama_model.n_heads
            ]
            v_caches_in = v_caches[
                offset : offset + self.layers_per_ctx * self.llama_model.n_heads
            ]
            export_inputs.append(k_caches_in)
            export_inputs.append(v_caches_in)

        return tuple(export_inputs)


def create_device_inputs(example_inputs, kv_input_numel, kv_type=torch.float32):
    # TODO: support batch inputs if necessary
    input_list = ""
    inputs, flat_inputs = [], []
    for input in example_inputs:
        if isinstance(input, list):
            for inp in input:
                flat_inputs.append(inp)
        else:
            flat_inputs.append(input)

    for i, data in enumerate(flat_inputs):
        input_list += f"input_0_{i}.raw "
        if data.flatten().shape[0] == kv_input_numel:
            data = data.to(dtype=kv_type)
        inputs.append(data)

    input_list += "\n"
    return tuple(inputs), input_list


if __name__ == "__main__":
    parser = setup_common_args_and_variables()
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. Default ./llama2_qnn",
        default="./llama2_qnn",
        type=str,
    )

    parser.add_argument(
        "-F",
        "--use_fp16",
        help="If specified, will run in fp16 precision and discard ptq setting",
        action="store_true",
        default=False,
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
        required=True,
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
        help="Pre-generated llama2.",
        type=str,
    )

    args = parser.parse_args()
    division = 4
    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)
    start_ts = time.time()
    with open(args.params) as f:
        config = ModelArgs(**json.load(f))
        # TODO: support batch inputs if necessary
        config.max_batch_size = 1
        config.max_seq_len = 1024
    device = "cpu"
    state_dict = torch.load(args.checkpoint, map_location=device, mmap=True)
    end_load_ts = time.time()
    print("torch.load checkpoint", end_load_ts - start_ts)
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
    end_load_state_dict_ts = time.time()
    print("instance.load_state_dict", end_load_state_dict_ts - end_load_ts)

    for l in llama_instance.layers:
        if getattr(l.attention, "prepare_sha", None):
            l.attention.prepare_sha()
    kv_type = torch.uint8
    if args.ptq == "8a8w":
        quant_dtype = QuantDtype.use_8a8w
    elif args.ptq == "16a4w":
        quant_dtype = QuantDtype.use_16a4w
    else:
        raise AssertionError(
            f"No support for quant type {args.ptq}. Support 8a8w and 16a4w."
        )

    if args.use_fp16:
        quant_dtype = None
    else:
        assert args.tokenizer_model is not None, "Need tokenizer model for calibration"

    if args.dtype_override is not None:
        dtype_override = DType[args.dtype_override]
        llama_instance = llama_instance.to(dtype_override.to_torch_dtype())

    llama_instance = convert_linear_to_conv2d(llama_instance)

    composite_llama = CompositeLlama(division, llama_instance.eval())
    kv_input_numel = (
        composite_llama.llama_meta["get_max_seq_len"] - 1
    ) * composite_llama.llama_meta["get_head_dim"]
    start_split_ts = time.time()
    inputs, input_list = create_device_inputs(
        composite_llama.get_export_inputs(), kv_input_numel, kv_type
    )
    pte_filename_list = []
    if args.pre_gen_pte is None:
        composite_llama.split_llama()
        end_split_ts = time.time()
        print("composite_llama.split_llama()", end_split_ts - start_split_ts)

        if quant_dtype is not None:
            composite_llama.quantize(
                quant_dtype,
                custom_annotations=(
                    annotate_matmul_16a8w,
                    annotate_linear_16a8w_in_affine_layer,
                ),
            )
            end_quantize_ts = time.time()
            print(
                "composite_llama.quantize(quant_dtype)", end_quantize_ts - end_split_ts
            )
        del llama_instance
        pte_filename_list = composite_llama.lowering_modules(
            args.artifact, kv_type=kv_type
        )
        assert len(pte_filename_list) != 0, "Failed to save pte file."
        end_lowering_ts = time.time()
        print("Complete Compile", end_lowering_ts - end_quantize_ts)
    else:
        for i in range(division + 2):
            pte_filename = f"llama2_qnn_{i}"
            pte_filename_list.append(f"{args.pre_gen_pte}/{pte_filename}.pte")

    workspace = f"/data/local/tmp/{getpass.getuser()}/executorch/composite_llama"
    pte_filenames = [Path(pte_filename).name for pte_filename in pte_filename_list]

    runner_args = " ".join(
        [
            f"--model_paths {','.join(pte_filenames)}",
            "--output_folder_path outputs",
            "--input_list_path input_list.txt",
            f"--tokenizer_path {os.path.basename(args.tokenizer_bin)}",
            f"--prompt {args.prompt}",
            f"--seq_len {args.seq_len}",
            f"--temperature {args.temperature}",
        ]
    )
    runner_cmd = " ".join(
        [
            f"cd {workspace} &&",
            "export ADSP_LIBRARY_PATH=. &&",
            "export LD_LIBRARY_PATH=. &&",
            f"./qnn_llama_runner {runner_args}",
        ]
    )

    if not args.compile_only:

        adb = SimpleADB(
            qnn_sdk=os.getenv("QNN_SDK_ROOT"),
            artifact_path=f"{args.build_folder}",
            pte_path=pte_filename_list,
            workspace=workspace,
            device_id=args.device,
            host_id=args.host,
            soc_model=args.model,
            shared_buffer=args.shared_buffer,
            runner="examples/qualcomm/qnn_llama_runner",
        )
        adb.push(inputs=[inputs], input_list=input_list, files=[args.tokenizer_bin])
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

        for idx, output in enumerate(outputs):
            print(f"Results[{idx}]:\n{output}")

    else:
        compile_only_dir = os.path.join(args.artifact, args.artifact)
        to_device_dir = os.path.join(compile_only_dir, "to_device")
        os.makedirs(to_device_dir, exist_ok=True)
        # input_list
        input_list_file = os.path.join(to_device_dir, "input_list.txt")
        with open(input_list_file, "w") as f:
            f.write(input_list)

        # write inputs
        for idx, data in enumerate([inputs]):
            flat_inputs = []
            for d in data:
                if isinstance(d, list):
                    for dd in d:
                        flat_inputs.append(dd)
                else:
                    flat_inputs.append(d)
            for i, d in enumerate(flat_inputs):
                filename = os.path.join(to_device_dir, f"input_{idx}_{i}.raw")
                d.detach().numpy().tofile(filename)

        # binaries
        arch_table = {
            "SM8650": "75",
            "SM8550": "73",
            "SM8475": "69",
            "SM8450": "69",
        }
        dsp_arch = arch_table[args.model]
        qnn_sdk_root = os.getenv("QNN_SDK_ROOT")

        on_device_files = [
            os.path.join(qnn_sdk_root, "lib", "aarch64-android", "libQnnHtp.so"),
            os.path.join(
                qnn_sdk_root,
                "lib",
                f"hexagon-v{dsp_arch}",
                "unsigned",
                f"libQnnHtpV{dsp_arch}Skel.so",
            ),
            os.path.join(
                qnn_sdk_root, "lib", "aarch64-android", f"libQnnHtpV{dsp_arch}Stub.so"
            ),
            os.path.join(qnn_sdk_root, "lib", "aarch64-android", "libQnnSystem.so"),
            os.path.join(args.build_folder, "examples", "qualcomm", "qnn_llama_runner"),
            os.path.join(
                args.build_folder,
                "backends",
                "qualcomm",
                "libqnn_executorch_backend.so",
            ),
        ] + pte_filename_list

        for on_device_file in on_device_files:
            shutil.copy2(on_device_file, to_device_dir)

        # tokenizer
        shutil.copy2(args.tokenizer_bin, to_device_dir)

        run_sh_lines = [
            "set -e",
            'SOURCEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"',
            f'adb_cmd="adb -s {args.device} -H {args.host}"',
            f'${{adb_cmd}} shell "rm -rf {workspace} && mkdir -p {workspace}/outputs"',
            f"${{adb_cmd}} push ${{SOURCEDIR}}/to_device/* {workspace}",
            f'${{adb_cmd}} shell "{runner_cmd}"',
            "echo",
            "echo ----- output_0_0.raw -----",
            "echo",
            f'${{adb_cmd}} shell "cat {workspace}/outputs/output_0_0.raw"',
            "",
        ]

        run_sh_file = os.path.join(compile_only_dir, "run.sh")
        with open(run_sh_file, "w") as fp:
            fp.write("\n".join(run_sh_lines))

        os.chmod(run_sh_file, stat.S_IRWXU | stat.S_IRWXG)

        print("Zipping files.....")
        shutil.make_archive(
            compile_only_dir,
            "zip",
            root_dir=args.artifact,
            base_dir=os.path.relpath(compile_only_dir, args.artifact),
        )

        print(f"Compile only mode, necessary files are written to {compile_only_dir}")
        print(f"And it's zipped as {compile_only_dir}.zip")
