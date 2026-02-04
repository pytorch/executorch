# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import inspect
import json
import os
import re
from functools import partial
from typing import Dict

import torch
from executorch.backends.qualcomm._passes import TagQuantIO
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_capture_program_passes,
)
from executorch.backends.qualcomm.builders.utils import is_graph_output
from executorch.backends.qualcomm.quantizer.annotators import Q_ANNOTATION_KEY
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.utils.constants import (
    QCOM_DTYPE,
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
    QCOM_QUANT_MAX,
    QCOM_QUANT_MIN,
    QCOM_SCALE,
    QCOM_ZERO_POINT,
)
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.devtools.backend_debug import print_delegation_info

from executorch.examples.qualcomm.oss_scripts.llama import LLMModelConfig

from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    ATTENTION_SINK_EVICTOR,
    DECODER_GRAPH_NAMES,
)
from executorch.examples.qualcomm.oss_scripts.llama.model.static_llama import (
    AttentionSinkRope,
    ModelArgs,
)
from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import (
    StaticLLMQuantRecipe,
)
from executorch.examples.qualcomm.oss_scripts.llama.wrappers.base_component import (
    Component,
    get_model_specific_kwargs,
    log_info,
    Mode,
    process_model_args,
    Processor,
    Request,
)
from executorch.examples.qualcomm.utils import make_quantizer
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from torch.fx import Node
from torchao.quantization.pt2e import FixedQParamsObserver, MinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer import QuantizationAnnotation, QuantizationSpec

dtype_map = {
    "torch.uint8": torch.uint8,
    "torch.int32": torch.int32,
}


def _annotate_zeros(gm: torch.fx.GraphModule) -> None:
    """
    Pop quant annotation for zero operator to avoid re-quantization
    """
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.zeros.default:
            node.meta.pop(Q_ANNOTATION_KEY)


def _annotate_kv_io(
    gm: torch.fx.GraphModule, kv_quant_attrs: Dict, kv_cache_shape: Dict
):
    """
    Overwrite the quantization annotation for KV cache IO to match the llama model's quantization annotation
    """

    input_index = 0
    for node in gm.graph.nodes:
        if node.op == "output":
            for index, kv_output in enumerate(node.args[0]):
                if kv_output.meta["val"].size() in kv_cache_shape.values():
                    kv_quant_attr = kv_quant_attrs[index]
                    fixed_observer = FixedQParamsObserver.with_args(
                        scale=kv_quant_attr[QCOM_SCALE],
                        zero_point=kv_quant_attr[QCOM_ZERO_POINT],
                        quant_min=kv_quant_attr[QCOM_QUANT_MIN],
                        quant_max=kv_quant_attr[QCOM_QUANT_MAX],
                        dtype=kv_quant_attr[QCOM_DTYPE],
                        qscheme=torch.torch.per_tensor_affine,
                    )

                    fixed_output_spec = QuantizationSpec(
                        quant_min=kv_quant_attr[QCOM_QUANT_MIN],
                        quant_max=kv_quant_attr[QCOM_QUANT_MAX],
                        dtype=kv_quant_attr[QCOM_DTYPE],
                        ch_axis=0,
                        observer_or_fake_quant_ctr=fixed_observer,
                    )

                    input_qspec_map = {}
                    for input in kv_output.args[0]:
                        if isinstance(input, Node):
                            input_qspec_map[input] = fixed_output_spec

                    kv_output.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                        input_qspec_map=input_qspec_map,
                        output_qspec=fixed_output_spec,
                        _annotated=True,
                    )
        elif node.op == "placeholder":
            if node.meta["val"].size() in kv_cache_shape.values():
                kv_quant_attr = kv_quant_attrs[input_index]
                fixed_observer = FixedQParamsObserver.with_args(
                    scale=kv_quant_attr[QCOM_SCALE],
                    zero_point=kv_quant_attr[QCOM_ZERO_POINT],
                    quant_min=kv_quant_attr[QCOM_QUANT_MIN],
                    quant_max=kv_quant_attr[QCOM_QUANT_MAX],
                    dtype=kv_quant_attr[QCOM_DTYPE],
                    qscheme=torch.torch.per_tensor_affine,
                )

                fixed_output_spec = QuantizationSpec(
                    quant_min=kv_quant_attr[QCOM_QUANT_MIN],
                    quant_max=kv_quant_attr[QCOM_QUANT_MAX],
                    dtype=kv_quant_attr[QCOM_DTYPE],
                    ch_axis=0,
                    observer_or_fake_quant_ctr=fixed_observer,
                )

                for user in node.users:
                    input_qspec_map = {}
                    for input in user.args:
                        if isinstance(input, Node):
                            input_qspec_map[input] = fixed_output_spec

                    user.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                        input_qspec_map=input_qspec_map,
                        output_qspec=fixed_output_spec,
                        _annotated=True,
                    )
                input_index += 1


def _calibration_data_generator(kv_quant_attrs: Dict, kcache_shape, vcache_shape):
    """
    Generate random tensors for KV cache with quantization parameter
    """

    while True:
        kcache_input_tensors = []
        vcache_input_tensors = []
        # Assume that half of the tensors are k cache and other half are v cache
        num_of_k_cache = len(kv_quant_attrs) // 2

        for index, kv_quant_attr in enumerate(kv_quant_attrs.values()):
            scale = kv_quant_attr[QCOM_SCALE]
            zp = kv_quant_attr[QCOM_ZERO_POINT]
            qmin = kv_quant_attr[QCOM_QUANT_MIN]
            qmax = kv_quant_attr[QCOM_QUANT_MAX]

            real_min = (qmin - zp) * scale
            real_max = (qmax - zp) * scale

            shape = kcache_shape if index < num_of_k_cache else vcache_shape

            input_tensor = torch.empty(shape).uniform_(real_min, real_max)
            if index < num_of_k_cache:
                kcache_input_tensors.append(input_tensor)
            else:
                vcache_input_tensors.append(input_tensor)

        yield (kcache_input_tensors, vcache_input_tensors)


def _parse_attention_sink_config(control_args: argparse.Namespace):
    """
    Parse attention sink config from command line args
    Args:
        control_args: Command line arguments
    Returns:
        sink_size: the number of initial tokens to keep as attention sink
        eviction_batch_size: the number of tokens to evict in batch when there is not enough space in the KV cache
    """
    assert control_args.use_attention_sink is not None
    attention_sink_params = control_args.use_attention_sink.split(",")
    # In the QNN backend, we are not using window arguments because currently, only static shape Llama is supported.
    assert len(attention_sink_params) == 2
    sink_size = int(attention_sink_params[0])
    eviction_batch_size = int(attention_sink_params[1])
    return sink_size, eviction_batch_size


def is_attention_sink_config_equal(
    attention_sink_evictor_pte_path: str, control_args: argparse.Namespace
):
    """
    Checks if the attention sink config in the PTE file matches the command line arguments.
    Returns True if they match, False otherwise.
    """
    sink_size, eviction_batch_size = _parse_attention_sink_config(control_args)

    assert os.path.exists(
        attention_sink_evictor_pte_path
    ), f"{attention_sink_evictor_pte_path} does not exist."
    with open(attention_sink_evictor_pte_path, "rb") as f:
        program_data = f.read()
    program = deserialize_pte_binary(program_data).program
    equal = True
    for method in program.execution_plan:
        if method.name == "get_eviction_batch_size":
            equal &= method.values[0].val.int_val == eviction_batch_size
        elif method.name == "get_sink_size":
            equal &= method.values[0].val.int_val == sink_size
        elif method.name == "get_max_context_len":
            equal &= method.values[0].val.int_val == control_args.max_context_len
    return equal


class AttentionSinkEvictor(Component):
    def __init__(
        self,
        control_args: argparse.Namespace,
        config: LLMModelConfig,
        mode: Mode,
    ):
        self.control_args = control_args
        self.config = config
        self.mode = mode
        quant_recipe: StaticLLMQuantRecipe = (
            self.config.quant_recipe(True) if self.config.quant_recipe else None
        )
        # load static llama model args
        params_path = (
            config.params_path if control_args.params is None else control_args.params
        )
        with open(params_path) as f:
            self.model_args = process_model_args(
                control_args, ModelArgs(**json.load(f)), quant_recipe, config, mode
            )

        self.evictor = self._prepare_model()
        self.passes_job = get_capture_program_passes()

    def _prepare_model(self) -> AttentionSinkRope:
        if self.mode == Mode.PREFILL and self.control_args.model_mode == "kv":
            return None

        sink_size, eviction_batch_size = _parse_attention_sink_config(self.control_args)

        assert (
            eviction_batch_size + sink_size <= self.model_args.max_context_len
        ), f"Please ensure sink_size({sink_size}) + eviction_batch_size({eviction_batch_size}) <= max_context_len({self.model_args.max_context_len})"
        assert (
            eviction_batch_size >= self.model_args.ar_len
        ), f"Please ensure eviction_batch_size({eviction_batch_size}) is >=_ar_len({self.model_args.ar_len})"
        assert (
            self.model_args.max_context_len > self.model_args.ar_len
        ), "The attention sink feature is not available for the Bert model."

        if self.control_args.model_mode == "lookahead":
            assert (
                eviction_batch_size >= self.control_args.ngram
            ), f"Please ensure eviction_batch_size({eviction_batch_size}) is >= ngram({self.control_args.ngram})"

        extra_kwargs = get_model_specific_kwargs(self.control_args, self.config)
        evictor = AttentionSinkRope(
            self.model_args,
            sink_size,
            eviction_batch_size,
            self.model_args.ar_len,
            **extra_kwargs,
        )
        self.meta = evictor.get_metadata()
        self.example_inputs = evictor.get_example_inputs()
        self.kv_cache_shape = evictor.kv_cache_shape
        return evictor

    def _tag_ios(self, node, fixed_point_type, kv_cache_shape: Dict):
        quant_io_type = None
        if (node.op == "placeholder" or is_graph_output(node)) and node.meta[
            "val"
        ].size() in kv_cache_shape.values():
            quant_io_type = fixed_point_type

        return quant_io_type

    @log_info
    def quantize(self, request: Request):
        if self.evictor is None:
            return
        fixed_point_type = torch.float32
        quant_dtype = None
        act_symmetric = False
        kv_io_bit_width = self.model_args.kv_io_bit_width
        if kv_io_bit_width == 8:
            fixed_point_type = torch.uint8
            quant_dtype = QuantDtype.use_8a8w
            act_symmetric = True
        elif kv_io_bit_width == 16:
            fixed_point_type = torch.uint16
            quant_dtype = QuantDtype.use_16a4w
        else:
            raise RuntimeError(f"Unknown kv io bit width {quant_dtype}")

        quantizer = make_quantizer(
            quant_dtype=quant_dtype,
            per_channel_conv=True,
            per_channel_linear=True,
            act_observer=MinMaxObserver,
            act_symmetric=act_symmetric,
        )
        custom_annotate_kv_io = request.method_data[
            ATTENTION_SINK_EVICTOR
        ].custom_annotation[0]
        quantizer.add_custom_quant_annotations(
            (
                partial(
                    custom_annotate_kv_io,
                    kv_cache_shape=self.kv_cache_shape,
                ),
                _annotate_zeros,
            )
        )
        num_data = 200
        data_generator = partial(
            request.method_data[ATTENTION_SINK_EVICTOR].calibration_data.datasets,
            kcache_shape=self.kv_cache_shape["k"],
            vcache_shape=self.kv_cache_shape["v"],
        )
        with torch.no_grad():
            self.evictor = torch.export.export(
                self.evictor, self.example_inputs, strict=True
            ).module()
            self.evictor = prepare_pt2e(self.evictor, quantizer)
            for _ in range(num_data):
                input = next(data_generator())
                self.evictor(*input)

            self.evictor = convert_pt2e(self.evictor)
        self.passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
        self.passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
            "get_quant_io_dtype_fn"
        ] = partial(
            self._tag_ios,
            fixed_point_type=fixed_point_type,
            kv_cache_shape=self.kv_cache_shape,
        )


class HybridAttentionSinkEvictor(Component):
    @log_info
    def __init__(self, control_args: argparse.Namespace, config: LLMModelConfig):
        self.decode_evictor = AttentionSinkEvictor(
            control_args,
            config,
            Mode.DECODE,
        )
        self.prefill_evictor = AttentionSinkEvictor(
            control_args,
            config,
            Mode.PREFILL,
        )
        self.control_args = control_args
        self.config = config
        self.set_next(self.decode_evictor).set_next(self.prefill_evictor)

    def process(self, request: Request) -> Request:
        Processor.process(self, request)

    @log_info
    def quantize(
        self,
        text_decoder_pte_path: str,
    ):
        assert os.path.exists(
            text_decoder_pte_path
        ), f"{text_decoder_pte_path} does not exist. Please compile the LLM first"
        with open(text_decoder_pte_path, "rb") as f:
            program_data = f.read()
        program = deserialize_pte_binary(program_data).program
        pat = re.compile(r"^get_kv_output_(?P<index>\d+)_quant_attr$")
        kv_quant_attrs = {}
        for method in program.execution_plan:
            if match := pat.fullmatch(method.name):
                kv_quant_attrs[int(match.group("index"))] = {
                    QCOM_SCALE: method.values[0].val.double_val,
                    QCOM_ZERO_POINT: method.values[1].val.int_val,
                    QCOM_QUANT_MIN: method.values[2].val.int_val,
                    QCOM_QUANT_MAX: method.values[3].val.int_val,
                    QCOM_DTYPE: dtype_map[method.values[4].val.string_val],
                }
        quantize_request = Request(
            inspect.currentframe().f_code.co_name,
            {
                ATTENTION_SINK_EVICTOR: Request.Data(
                    custom_annotation=(
                        partial(
                            _annotate_kv_io,
                            kv_quant_attrs=kv_quant_attrs,
                        ),
                    ),
                    calibration_data=Request.CalibrationData(
                        datasets=partial(
                            _calibration_data_generator, kv_quant_attrs=kv_quant_attrs
                        ),
                    ),
                )
            },
        )
        self.process(quantize_request)

    @log_info
    def compile(self, attention_sink_evictor_pte_path: str):
        models = [
            d
            for d in [self.decode_evictor, self.prefill_evictor]
            if d.evictor is not None
        ]
        # To align with LLM model, we keep the graph name as forward when using kv mode for evaluation LLM models
        graph_names = DECODER_GRAPH_NAMES[: len(models)]
        backend_options = generate_htp_compiler_spec(use_fp16=False)
        compiler_specs = [
            generate_qnn_executorch_compiler_spec(
                soc_model=get_soc_to_chipset_map()[self.control_args.model],
                backend_options=backend_options,
                shared_buffer=not self.control_args.enable_x86_64,
            )
        ] * len(models)

        edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
            module=dict(zip(graph_names, [model.evictor for model in models])),
            inputs=dict(zip(graph_names, [model.example_inputs for model in models])),
            compiler_specs=dict(zip(graph_names, compiler_specs)),
            constant_methods=self.decode_evictor.meta,
            passes_job=dict(zip(graph_names, [model.passes_job for model in models])),
        )

        for ep in edge_prog_mgr._edge_programs.values():
            print_delegation_info(ep.graph_module)

        executorch_config = ExecutorchBackendConfig(
            # For shared buffer, user must pass the memory address
            # which is allocated by RPC memory to executor runner
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False,
                alloc_graph_output=False,
            ),
            extract_delegate_segments=True,
        )
        exec_prog_mgr = edge_prog_mgr.to_executorch(executorch_config)
        with open(attention_sink_evictor_pte_path, "wb") as file:
            exec_prog_mgr.write_to_file(file)
