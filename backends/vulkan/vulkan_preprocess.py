# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
from typing import Dict, final, List

import executorch.backends.vulkan.serialization.vulkan_graph_schema as vk_graph_schema
from executorch.backends.vulkan.serialization.vulkan_graph_serialize import (
    convert_to_flatbuffer,
)

from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    ExportedProgram,
    PreprocessResult,
)

from executorch.exir.passes import MemoryPlanningPass, SpecPropPass

from executorch.exir.program._program import _copy_module
from executorch.exir.tensor import TensorSpec
from torch import dtype, float32
from torch.fx import Node
from torch.fx.node import Argument

DEFAULT_DEBUG_HANDLE = 65535


@final
class VulkanBackend(BackendDetails):
    @staticmethod
    def get_vk_op_type(
        target_name: str, kwargs: Dict[str, "Argument"]
    ) -> vk_graph_schema.VkArithmeticOpType:
        if target_name == "aten.add.Tensor":
            return vk_graph_schema.VkArithmeticOpType.vk_arithmetic_op_type_add
        elif target_name == "aten.sub.Tensor":
            return vk_graph_schema.VkArithmeticOpType.vk_arithmetic_op_type_sub
        elif target_name == "aten.mul.Tensor":
            return vk_graph_schema.VkArithmeticOpType.vk_arithmetic_op_type_mul
        elif target_name == "aten.div.Tensor":
            return vk_graph_schema.VkArithmeticOpType.vk_arithmetic_op_type_div
        elif target_name == "aten.div.Tensor_mode":
            if kwargs.get("rounding_mode", None) == "floor":
                return (
                    vk_graph_schema.VkArithmeticOpType.vk_arithmetic_op_type_floor_div
                )

            raise AssertionError(
                f"Invalid node kwargs for vulkan_preprocess (target_name: {target_name}, "
                f"kwargs: {kwargs})"
            )
        elif target_name == "aten.pow.Tensor_Tensor":
            return vk_graph_schema.VkArithmeticOpType.vk_arithmetic_op_type_pow

        else:
            raise AssertionError(
                f"Invalid node target name for vulkan_preprocess ({target_name})"
            )

    @staticmethod
    def get_vk_datatype(torch_dtype: dtype) -> vk_graph_schema.VkDataType:
        if torch_dtype == float32:
            return vk_graph_schema.VkDataType.fp32
        else:
            raise AssertionError(f"Invalid dtype for vulkan_preprocess ({torch_dtype})")

    @classmethod
    # pyre-ignore
    def preprocess(  # noqa: C901
        cls,
        program: ExportedProgram,
        module_compile_spec: List[CompileSpec],
    ) -> PreprocessResult:
        vk_nodes = []
        vk_values = []
        vk_const_buffers = [vk_graph_schema.Buffer(storage=bytes())]
        vk_input_ids = []
        vk_output_ids = []

        # Map from graph Node to schema VkValue.
        node_to_value_ids = {}

        def create_single_vk_value(node: Node, buffer_id: int = 0) -> int:
            spec = node.meta.get("spec")
            assert isinstance(spec, TensorSpec)
            new_id = len(vk_values)
            if node not in node_to_value_ids:
                node_to_value_ids[node] = new_id
            else:
                current_ids = node_to_value_ids[node]
                if isinstance(current_ids, int):
                    current_ids = [current_ids, new_id]
                else:
                    current_ids.append(new_id)

            # Negative id indicates that this tensor will have its own dedicated memory.
            mem_obj_id = -1
            if spec.mem_obj_id is not None:
                mem_obj_id = spec.mem_obj_id

            vk_values.append(
                vk_graph_schema.VkValue(
                    value=vk_graph_schema.VkTensor(
                        datatype=VulkanBackend.get_vk_datatype(spec.dtype),
                        dims=spec.shape,
                        constant_buffer_id=buffer_id,
                        mem_obj_id=mem_obj_id,
                    )
                )
            )
            return new_id

        def create_vk_values_for(node: Node, buffer_id: int = 0):
            spec = node.meta.get("spec")

            if isinstance(spec, TensorSpec):
                return create_single_vk_value(node, buffer_id)
            else:
                ids = []
                for _ in spec:
                    ids.append(create_single_vk_value(node, buffer_id))
                return ids

        passes = [
            SpecPropPass(),
            MemoryPlanningPass("greedy"),
        ]

        new_gm = program.graph_module
        for p in passes:
            # This is a workaround to allow the memory planning pass to work without
            # having to first apply ToOutVarPass(). See the `greedy()` function in
            # `exir.memory_planning`; if this attribute isn't set, assertions in
            # `collect_spec_from_nodes()` will fail.
            if isinstance(p, MemoryPlanningPass):
                new_gm.encounter_to_out_var_failure = True
            new_gm_res = p(new_gm)
            assert new_gm_res is not None
            new_gm = new_gm_res.graph_module
        _copy_module(program.graph_module, new_gm)

        for node in program.graph_module.graph.nodes:
            if node.op == "placeholder":
                # Input
                ids = create_vk_values_for(node)
                if isinstance(ids, int):
                    vk_input_ids.append(ids)
                else:
                    vk_input_ids += ids
            elif node.op == "call_function":
                # Op
                if (
                    node.all_input_nodes[0] not in node_to_value_ids
                    or node.all_input_nodes[1] not in node_to_value_ids
                ):
                    raise AssertionError(
                        "Cannot find input(s) for current node in node_to_value_ids. This means this node is being serialized before its input(s) which is not allowed."
                    )
                vk_nodes.append(
                    vk_graph_schema.VkNode(
                        node=vk_graph_schema.VkArithmeticNode(
                            input1_id=node_to_value_ids[node.all_input_nodes[0]],
                            input2_id=node_to_value_ids[node.all_input_nodes[1]],
                            output_id=create_vk_values_for(node),
                            op_type=VulkanBackend.get_vk_op_type(
                                target_name=node.target.__name__, kwargs=node.kwargs
                            ),
                            flags=0,
                        ),
                        debug_handle=node.meta.get(
                            "debug_handle", DEFAULT_DEBUG_HANDLE
                        ),
                    ),
                )
            elif node.op == "get_attr":
                # Adapted from https://fburl.com/code/adyy0m6x
                buffer_id = len(vk_const_buffers)

                const_val = getattr(node.graph.owning_module, node.target).contiguous()
                # pyre-ignore
                array_type = ctypes.c_char * const_val.untyped_storage().nbytes()
                array = ctypes.cast(
                    const_val.untyped_storage().data_ptr(),
                    ctypes.POINTER(array_type),
                ).contents
                buffer = vk_graph_schema.Buffer(storage=bytes(array))
                vk_const_buffers.append(buffer)

                create_vk_values_for(node, buffer_id)

            elif node.op == "output":
                if node.all_input_nodes[0] not in node_to_value_ids:
                    raise AssertionError(
                        "Cannot find input to output node in node_to_value_ids. This means the output node is being serialized before its corresponding internal node which is not allowed."
                    )
                vk_output_ids.append(node_to_value_ids[node.all_input_nodes[0]])
            else:
                raise RuntimeError(f"Unsupported op, {node.op}, in Vulkan Preprocess")
        vk_graph = vk_graph_schema.VkGraph(
            version="0",
            nodes=vk_nodes,
            values=vk_values,
            constant_buffers=vk_const_buffers,
            input_ids=vk_input_ids,
            output_ids=vk_output_ids,
        )
        return PreprocessResult(
            processed_bytes=convert_to_flatbuffer(vk_graph),
        )
