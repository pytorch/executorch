from typing import final, List

import executorch.backends.vulkan.serialization.vulkan_graph_schema as vk_graph_schema

# pyre-ignore Undefined import [21]: Could not find a module corresponding to import `executorch.exir.bindings`
import executorch.exir.bindings as bindings  # @manual=//executorch/exir:bindings

from executorch.backends.backend_details import BackendDetails, CompileSpec
from executorch.backends.vulkan.serialization.vulkan_graph_serialize import (
    convert_to_flatbuffer,
)
from executorch.exir import ExportGraphModule
from torch import dtype, float32, Tensor
from torch.fx import Node

DEFAULT_DEBUG_HANDLE = 65535


@final
class VulkanBackend(BackendDetails):
    @staticmethod
    def get_vk_op_type(target_name: str) -> vk_graph_schema.VkArithmeticOpType:
        if target_name == "add.Tensor":
            return vk_graph_schema.VkArithmeticOpType.vk_arithmetic_op_type_add
        elif target_name == "sub.Tensor":
            return vk_graph_schema.VkArithmeticOpType.vk_arithmetic_op_type_sub
        elif target_name == "mul.Tensor":
            return vk_graph_schema.VkArithmeticOpType.vk_arithmetic_op_type_mul
        elif target_name == "div.Tensor":
            return vk_graph_schema.VkArithmeticOpType.vk_arithmetic_op_type_div
        else:
            raise AssertionError(
                f"Invalid node target name for vulkan_preprocess ({target_name})"
            )

    @staticmethod
    def get_vk_datatype(torch_dtype: dtype) -> vk_graph_schema.VkDatatype:
        if torch_dtype == float32:
            return vk_graph_schema.VkDatatype.vk_datatype_fp32
        else:
            raise AssertionError(f"Invalid dtype for vulkan_preprocess ({torch_dtype})")

    @classmethod
    # pyre-ignore
    def preprocess(
        cls,
        edge_ir_module: ExportGraphModule,
        module_compile_spec: List[CompileSpec],
    ) -> bytes:
        vk_nodes = []
        vk_values = []
        vk_input_ids = []
        vk_output_ids = []
        vk_const_buffers = [vk_graph_schema.Buffer(storage=bytes())]

        # Mapping from node in the executorch graph to corresponding VkValue id
        node_vk_value_ids = {}

        def assign_vk_value_id(node: Node, tensor: Tensor, buffer_idx: int) -> int:
            assert node not in node_vk_value_ids
            new_id = len(vk_values)
            node_vk_value_ids[node] = new_id
            vk_values.append(
                vk_graph_schema.VkValue(
                    value=vk_graph_schema.VkTensor(
                        datatype=VulkanBackend.get_vk_datatype(tensor.dtype),
                        dims=list(tensor.shape),
                        constant_buffer_idx=buffer_idx,
                    )
                )
            )
            return new_id

        def assign_non_const_vk_value_id(node: Node) -> int:
            return assign_vk_value_id(node, node.meta["val"], 0)

        for node in edge_ir_module.graph.nodes:
            if node.op == "placeholder":
                # Input
                vk_input_ids.append(assign_non_const_vk_value_id(node))
            elif node.op == "call_function":
                # Op
                if (
                    node.all_input_nodes[0] not in node_vk_value_ids
                    or node.all_input_nodes[1] not in node_vk_value_ids
                ):
                    raise AssertionError(
                        "Cannot find input(s) for current node in node_vk_value_ids. This means this node is being serialized before its input(s) which is not allowed."
                    )
                vk_nodes.append(
                    vk_graph_schema.VkNode(
                        node=vk_graph_schema.VkArithmeticNode(
                            input1_id=node_vk_value_ids[node.all_input_nodes[0]],
                            input2_id=node_vk_value_ids[node.all_input_nodes[1]],
                            output_id=assign_non_const_vk_value_id(node),
                            op_type=VulkanBackend.get_vk_op_type(node.target.__name__),
                            flags=0,
                        ),
                        debug_handle=node.meta.get(
                            "debug_handle", DEFAULT_DEBUG_HANDLE
                        ),
                    ),
                )
            elif node.op == "get_attr":
                # Tensor
                # Adapted from https://www.internalfb.com/code/fbsource/[18c174b709f321d26e6632e2f826498cde730f8c]/fbcode/executorch/backends/xnnpack/xnnpack_preprocess.py?lines=127
                buffer_idx = len(vk_const_buffers)

                const_val = getattr(node.graph.owning_module, node.target).contiguous()
                buffer = vk_graph_schema.Buffer(
                    # pyre-ignore[16]: Module executorch.exir has no attribute bindings.
                    storage=bindings.copy_buffer(
                        const_val.untyped_storage().data_ptr(),
                        const_val.untyped_storage().nbytes(),
                    )
                )
                vk_const_buffers.append(buffer)

                assign_vk_value_id(node, const_val, buffer_idx)

            elif node.op == "output":
                if node.all_input_nodes[0] not in node_vk_value_ids:
                    raise AssertionError(
                        "Cannot find input to output node in node_vk_value_ids. This means the output node is being serialized before its corresponding internal node which is not allowed."
                    )
                vk_output_ids.append(node_vk_value_ids[node.all_input_nodes[0]])
            else:
                raise RuntimeError(f"Unsupported op, {node.op}, in Vulkan Preprocess")
        vk_graph = vk_graph_schema.VkGraph(
            version="0",
            vknodes=vk_nodes,
            vkvalues=vk_values,
            input_ids=vk_input_ids,
            output_ids=vk_output_ids,
            constant_buffer=vk_const_buffers,
        )
        return convert_to_flatbuffer(vk_graph)
