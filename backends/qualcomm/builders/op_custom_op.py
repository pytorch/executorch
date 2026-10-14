# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import Dict, Iterable, List, Optional

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager

import numpy as np
import torch

from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchOpPackageInfo,
)

from executorch.backends.qualcomm.utils.constants import (
    QCOM_AXIS_ORDER,
    QCOM_CUSTOM_OP_INPUT_AXIS_ORDERS,
    QCOM_CUSTOM_OP_OUTPUT_AXIS_ORDERS,
    QCOM_DATA,
)

from .node_visitor import NodeVisitor, QNN_TENSOR_TYPE_MAP
from .qnn_constants import OpTranspose, QNN_OP_PACKAGE_NAME_QTI_AISW
from .utils import (
    get_custom_op_output_source,
    is_graph_input,
    uses_compatible_axis_order,
)


def _resolve_qnn_data_type(
    py_type: type, arg_name: str, target: str, *, of_element: bool = False
):
    """QNN data type for a custom-op arg type, or a ValueError naming the arg."""
    what = "element type" if of_element else "type"
    if py_type is str:
        raise ValueError(
            f"String {what} for argument '{arg_name}' of {target} is unsupported: "
            "QNN_DATATYPE_STRING is not plumbed through AddScalarParam in "
            "aot/python/PyQnnManagerAdaptor.h. Encode it as an int in the op "
            "schema, or add the missing case."
        )
    if py_type not in QNN_TENSOR_TYPE_MAP:
        raise ValueError(
            f"Argument '{arg_name}' of {target} has unsupported {what} "
            f"{py_type.__name__}: QNN_TENSOR_TYPE_MAP has no entry for it."
        )
    return QNN_TENSOR_TYPE_MAP[py_type]


class CustomOp(NodeVisitor):
    target = ""
    op_package_info = QnnExecuTorchOpPackageInfo()

    def __init__(
        self,
        op_package_info: QnnExecuTorchOpPackageInfo,
        *args,
    ) -> None:
        super().__init__(*args)
        self.target = op_package_info.custom_op_name
        self.op_package_info = op_package_info

    @staticmethod
    def _validate_output_axis_orders(
        node: torch.fx.Node, output_axis_orders: Dict[int, tuple[int, ...]]
    ) -> None:
        outputs = node.meta["val"]
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        for index, axis_order in output_axis_orders.items():
            if (
                index < 0
                or index >= len(outputs)
                or outputs[index] is None
                or tuple(sorted(axis_order)) != tuple(range(outputs[index].dim()))
            ):
                raise ValueError(
                    f"Custom op {node.name} has invalid output {index} axis order "
                    f"{axis_order}."
                )

    @staticmethod
    def _validate_input_axis_orders(
        node: torch.fx.Node, input_axis_orders: Dict[int, tuple[int, ...]]
    ) -> None:
        for index, axis_order in input_axis_orders.items():
            if (
                index < 0
                or index >= len(node.args)
                or not isinstance(node.args[index], torch.fx.Node)
            ):
                raise ValueError(
                    f"Custom op {node.name} has invalid input {index} axis order "
                    f"{axis_order}."
                )
        for index, arg in enumerate(node.args):
            if not isinstance(arg, torch.fx.Node):
                continue
            input_val = arg.meta.get("val")
            if input_val is None:
                raise ValueError(
                    f"Custom op {node.name} input {index} has no tensor metadata."
                )
            axis_order = input_axis_orders.get(index)
            if axis_order is not None and tuple(sorted(axis_order)) != tuple(
                range(input_val.dim())
            ):
                raise ValueError(
                    f"Custom op {node.name} has invalid input {index} axis order "
                    f"{axis_order}."
                )

    def _validate_shared_input_axis_order(
        self,
        input_node: torch.fx.Node,
        axis_order: tuple[int, ...],
        nodes_to_wrappers: Dict[str, Dict[int, PyQnnManager.TensorWrapper]],
        custom_node: torch.fx.Node,
        input_rank: int,
    ) -> None:
        """Reject layouts that require reusing one cached QNN tensor differently."""
        identity_axis_order = tuple(range(input_rank))
        if axis_order != identity_axis_order and any(
            user is not custom_node
            and not uses_compatible_axis_order(user, input_node, axis_order)
            for user in input_node.users
        ):
            raise ValueError(
                f"Custom op input {input_node.name} has an incompatible consumer; "
                "an explicit QNN transpose is required."
            )
        cached = nodes_to_wrappers.get(input_node.name, {}).get(0)
        if cached is None:
            return

        layout_node, output_index = get_custom_op_output_source(input_node)
        cached_layout = self._wrapper_axis_orders.get((layout_node.name, output_index))
        cached_axis_order = (
            cached_layout[1]
            if cached_layout is not None and cached_layout[0] is cached
            else None
        )
        if cached_axis_order is None:
            cached_axis_order = layout_node.meta.get(
                QCOM_CUSTOM_OP_OUTPUT_AXIS_ORDERS, {}
            ).get(output_index)
        if cached_axis_order is None:
            cached_axis_order = input_node.meta.get(QCOM_AXIS_ORDER)
        if cached_axis_order is None:
            cached_axis_order = layout_node.meta.get(QCOM_AXIS_ORDER)
        if cached_axis_order is None:
            cached_axis_order = identity_axis_order
        if tuple(cached_axis_order) != axis_order:
            raise ValueError(
                f"Custom op input {input_node.name} has conflicting axis orders "
                f"{tuple(cached_axis_order)} and {axis_order}; an explicit QNN "
                "transpose is required."
            )

    def _define_graph_input_axis_transform(
        self,
        input_node: torch.fx.Node,
        node: torch.fx.Node,
        input_tensor: torch.Tensor,
        axis_order: tuple[int, ...],
        nodes_to_wrappers: Dict[str, Dict[int, PyQnnManager.TensorWrapper]],
    ) -> tuple[PyQnnManager.TensorWrapper, Optional[PyQnnManager.PyQnnOpWrapper]]:
        input_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        if axis_order == tuple(range(input_tensor.dim())):
            return input_wrapper, None
        transformed_tensor = input_tensor.permute(axis_order).contiguous()
        transform_name = f"{input_node.name}_to_{'_'.join(map(str, axis_order))}"
        if transformed_wrapper := nodes_to_wrappers.get(transform_name, {}).get(0):
            return transformed_wrapper, None
        quant_encoding, quant_configs = self.get_quant_encoding_conf(
            input_node, node, axis_order
        )
        transformed_wrapper = self.define_custom_tensor_wrapper(
            node_name=transform_name,
            tensor_type=PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            dtype=PyQnnManager.PyQnnTensorWrapper(input_wrapper).GetDataType(),
            quant_encoding=quant_encoding,
            quant_configs=quant_configs,
            dims=transformed_tensor.size(),
            tensor=transformed_tensor,
            is_fake_tensor=True,
            nodes_to_wrappers=nodes_to_wrappers,
        )
        transform_op = PyQnnManager.PyQnnOpWrapper(
            transform_name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpTranspose.op_name,
        )
        transform_op.AddInputTensors([input_wrapper])
        transform_op.AddOutputTensors([transformed_wrapper])
        transform_op.AddTensorParam(
            OpTranspose.param_perm,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_UINT_32,
            1,
            [len(axis_order)],
            np.array(axis_order, dtype=np.uint32),
            True,
        )
        return transformed_wrapper, transform_op

    def _define_output_tensors(
        self,
        node: torch.fx.Node,
        output_axis_orders: Dict[int, tuple[int, ...]],
        nodes_to_wrappers: Dict[str, Dict[int, PyQnnManager.TensorWrapper]],
    ) -> List[PyQnnManager.TensorWrapper]:
        outputs = node.meta["val"]
        output_count = len(outputs) if isinstance(outputs, tuple) else 1
        output_tensors = []
        for index in range(output_count):
            output_tensor = self.get_tensor(
                node, node, index if output_count > 1 else None
            )
            axis_order = output_axis_orders.get(index)
            if axis_order is not None:
                output_tensor = output_tensor.permute(axis_order).contiguous()
            output_tensor_wrapper = self.define_tensor(
                node,
                node,
                output_tensor,
                PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                nodes_to_wrappers,
                wrapper_idx=index,
                axis_order=axis_order,
            )
            output_tensors.append(output_tensor_wrapper)
            self._wrapper_axis_orders[(node.name, index)] = (
                output_tensor_wrapper,
                tuple(output_axis_orders.get(index, range(output_tensor.dim()))),
            )
        return output_tensors

    def _add_attribute_parameter(
        self,
        custom_op: PyQnnManager.PyQnnOpWrapper,
        arg_name: str,
        arg,
    ) -> None:
        if isinstance(arg, str):
            # str is Iterable, so it would otherwise be taken for a tensor param.
            _resolve_qnn_data_type(str, arg_name, self.target)
        elif isinstance(arg, Iterable):
            values = list(arg)
            if not values:
                raise ValueError(
                    f"Argument '{arg_name}' of {self.target} is empty: a QNN "
                    "tensor param needs at least one element."
                )
            tensor_parm_shape = [len(values)]
            custom_op.AddTensorParam(
                arg_name,
                _resolve_qnn_data_type(
                    type(values[0]), arg_name, self.target, of_element=True
                ),
                len(tensor_parm_shape),
                tensor_parm_shape,
                np.array(values),
                True,
            )
        else:
            custom_op.AddScalarParam(
                arg_name,
                _resolve_qnn_data_type(type(arg), arg_name, self.target),
                {QCOM_DATA: arg},
            )

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[str, Dict[int, PyQnnManager.TensorWrapper]],
    ) -> PyQnnManager.PyQnnOpWrapper | List[PyQnnManager.PyQnnOpWrapper] | None:
        custom_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            self.op_package_info.op_package_name,
            self.op_package_info.qnn_op_type_name,
        )
        input_axis_orders = node.meta.get(QCOM_CUSTOM_OP_INPUT_AXIS_ORDERS, {})
        output_axis_orders = node.meta.get(QCOM_CUSTOM_OP_OUTPUT_AXIS_ORDERS, {})
        self._validate_output_axis_orders(node, output_axis_orders)

        custom_input_tensors = []
        custom_attr_keys = [arg.name for arg in node.target._schema.arguments]
        if len(custom_attr_keys) != len(node.args):
            warnings.warn(
                f"Number of inputs ({len(node.args)}) mismatch the number of args ({len(custom_attr_keys)}) in schema for the custom node ({self.target}).",
                stacklevel=1,
            )
            return
        self._validate_input_axis_orders(node, input_axis_orders)
        input_transform_ops = []
        for i, (arg, arg_name) in enumerate(zip(node.args, custom_attr_keys)):
            if arg is None:
                continue
            if isinstance(arg, torch.fx.Node):
                input_val = arg.meta["val"]
                input_rank = input_val.dim()
                axis_order = tuple(input_axis_orders.get(i, range(input_rank)))
                is_graph_input_node = is_graph_input(arg, self.edge_program)
                if not is_graph_input_node:
                    self._validate_shared_input_axis_order(
                        arg, axis_order, nodes_to_wrappers, node, input_rank
                    )
                input_tensor = self.get_tensor(arg, node)
                if is_graph_input_node and i in input_axis_orders:
                    input_tensor_wrapper, transform_op = (
                        self._define_graph_input_axis_transform(
                            arg,
                            node,
                            input_tensor,
                            axis_order,
                            nodes_to_wrappers,
                        )
                    )
                    if transform_op is not None:
                        input_transform_ops.append(transform_op)
                else:
                    if i in input_axis_orders:
                        input_tensor = input_tensor.permute(axis_order).contiguous()
                    input_tensor_wrapper = self.define_tensor(
                        arg,
                        node,
                        input_tensor,
                        PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                        nodes_to_wrappers,
                        axis_order=input_axis_orders.get(i),
                    )
                if not is_graph_input_node:
                    self._wrapper_axis_orders[(arg.name, 0)] = (
                        input_tensor_wrapper,
                        axis_order,
                    )
                custom_input_tensors.append(input_tensor_wrapper)
            else:
                self._add_attribute_parameter(custom_op, arg_name, arg)

        custom_output_tensors = self._define_output_tensors(
            node, output_axis_orders, nodes_to_wrappers
        )

        custom_op.AddInputTensors(custom_input_tensors)
        custom_op.AddOutputTensors(custom_output_tensors)
        return input_transform_ops + [custom_op] if input_transform_ops else custom_op
