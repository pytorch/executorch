# Copyright 2026 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import Any, cast, List

import torch
import tosa_serializer as ts  # type: ignore
from executorch.backends.arm.operators.node_visitor import (
    NodeVisitor,
    register_node_visitor,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.constant_pool import TosaSerializerWithConstantPool
from executorch.backends.arm.tosa.mapping import TosaArg
from executorch.backends.arm.tosa.utils import normalize_symint


@register_node_visitor
class TosaConstShapeVisitor(NodeVisitor):
    target = "tosa.CONST_SHAPE.default"

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        shape_input = inputs[0].special
        rank = len(shape_input)
        vals = normalize_symint(node.meta["val"])
        tosa_graph = cast(TosaSerializerWithConstantPool, tosa_graph)
        # Downstream visitors reference this FX output by name. Pooling it with a
        # serializer-generated constant could leave output.name undefined.
        tosa_graph.addUnpooledConst(
            [
                rank,
            ],
            dtype=ts.DType.SHAPE,  # pyrefly: ignore [missing-attribute]
            vals=vals,
            name=output.name,
        )


class TosaShapeNodeVisitor(NodeVisitor):

    tosa_specs = TosaSpecification.all_profiles_for_version("1.1")

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        tosa_graph = cast(ts.TosaSerializer, tosa_graph)  # pyrefly: ignore [missing-attribute]
        tosa_graph.currRegion.currBasicBlock.addShape(
            output.name,
            output.shape[0],  # pyrefly: ignore [unsupported-operation]
        )


class TosaBasicShapeVisitor(TosaShapeNodeVisitor):
    tosa_op: ts.Op  # pyrefly: ignore [missing-attribute]
    attr_method: str

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        super().define_node(node, tosa_graph, inputs, output)
        self.serialize(
            node,
            tosa_graph,
            tosa_op=self.tosa_op,
            inputs=inputs,
            output=output,
            attr_method=self.attr_method,
        )


@register_node_visitor
class TosaDimShapeVisitor(TosaShapeNodeVisitor):
    target = "tosa.DIM.default"

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        super().define_node(node, tosa_graph, inputs, output)

        attr = ts.TosaSerializerAttribute()  # pyrefly: ignore [missing-attribute]
        attr.DimAttribute(axis=node.kwargs["axis"])
        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.DIM,  # pyrefly: ignore [missing-attribute]
            [inputs[0].name],
            [output.name],
            attr,
        )


@register_node_visitor
class TosaAddShapeVisitor(TosaBasicShapeVisitor):
    target = "tosa.ADD_SHAPE.default"

    tosa_op = ts.Op.ADD_SHAPE  # pyrefly: ignore [missing-attribute]
    attr_method = "AddShapeAttribute"


@register_node_visitor
class TosaSubShapeVisitor(TosaBasicShapeVisitor):
    target = "tosa.SUB_SHAPE.default"

    tosa_op = ts.Op.SUB_SHAPE  # pyrefly: ignore [missing-attribute]
    attr_method = "SubShapeAttribute"


@register_node_visitor
class TosaAssertEqualShapeVisitor(TosaShapeNodeVisitor):
    target = "tosa.ASSERT_EQUAL_SHAPE.default"

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        super().define_node(node, tosa_graph, inputs, output)
        tosa_graph = cast(ts.TosaSerializer, tosa_graph)  # pyrefly: ignore [missing-attribute]
        attr = ts.TosaSerializerAttribute()  # pyrefly: ignore [missing-attribute]
        attr.AssertEqualShapeAttribute(allow_broadcast=node.kwargs["allow_broadcast"])
        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.ASSERT_EQUAL_SHAPE,  # pyrefly: ignore [missing-attribute]
            [inputs[0].name, inputs[1].name],
            [output.name],
            attr,
        )


@register_node_visitor
class TosaCatShapeVisitor(TosaShapeNodeVisitor):
    target = "tosa.CONCAT_SHAPE.default"

    def define_node(
        self,
        node: torch.fx.Node,
        tosa_graph: Any,
        inputs: List[TosaArg],
        output: TosaArg,
    ) -> None:
        super().define_node(node, tosa_graph, inputs, output)
        tosa_graph = cast(ts.TosaSerializer, tosa_graph)  # pyrefly: ignore [missing-attribute]

        input_shape_list = [input.name for input in inputs[0].special]

        attr = ts.TosaSerializerAttribute()  # pyrefly: ignore [missing-attribute]
        attr.ConcatShapeAttribute()
        self._serialize_operator(
            node,
            tosa_graph,
            ts.Op.CONCAT_SHAPE,  # pyrefly: ignore [missing-attribute]
            input_shape_list,
            [output.name],
            attr,
        )


@register_node_visitor
class TosaDivCeilShapeVisitor(TosaBasicShapeVisitor):
    target = "tosa.DIV_CEIL_SHAPE.default"

    tosa_op = ts.Op.DIV_CEIL_SHAPE  # pyrefly: ignore [missing-attribute]
    attr_method = "DivCeilShapeAttribute"


@register_node_visitor
class TosaDivShapeVisitor(TosaBasicShapeVisitor):
    target = "tosa.DIV_FLOOR_SHAPE.default"

    tosa_op = ts.Op.DIV_FLOOR_SHAPE  # pyrefly: ignore [missing-attribute]
    attr_method = "DivFloorShapeAttribute"


@register_node_visitor
class TosaExp2ShapeVisitor(TosaBasicShapeVisitor):
    target = "tosa.EXP2_SHAPE.default"

    tosa_op = ts.Op.EXP2_SHAPE  # pyrefly: ignore [missing-attribute]
    attr_method = "Exp2ShapeAttribute"


@register_node_visitor
class TosaLog2CeilShapeVisitor(TosaBasicShapeVisitor):
    target = "tosa.LOG2_CEIL_SHAPE.default"

    tosa_op = ts.Op.LOG2_CEIL_SHAPE  # pyrefly: ignore [missing-attribute]
    attr_method = "Log2CeilShapeAttribute"


@register_node_visitor
class TosaLog2FloorShapeVisitor(TosaBasicShapeVisitor):
    target = "tosa.LOG2_FLOOR_SHAPE.default"

    tosa_op = ts.Op.LOG2_FLOOR_SHAPE  # pyrefly: ignore [missing-attribute]
    attr_method = "Log2FloorShapeAttribute"


@register_node_visitor
class TosaMaxShapeVisitor(TosaBasicShapeVisitor):
    target = "tosa.MAX_SHAPE.default"

    tosa_op = ts.Op.MAX_SHAPE  # pyrefly: ignore [missing-attribute]
    attr_method = "MaxShapeAttribute"


@register_node_visitor
class TosaMinShapeVisitor(TosaBasicShapeVisitor):
    target = "tosa.MIN_SHAPE.default"

    tosa_op = ts.Op.MIN_SHAPE  # pyrefly: ignore [missing-attribute]
    attr_method = "MinShapeAttribute"


@register_node_visitor
class TosaMulShapeVisitor(TosaBasicShapeVisitor):
    target = "tosa.MUL_SHAPE.default"

    tosa_op = ts.Op.MUL_SHAPE  # pyrefly: ignore [missing-attribute]
    attr_method = "MulShapeAttribute"


@register_node_visitor
class TosaSliceShapeVisitor(TosaBasicShapeVisitor):
    target = "tosa.SLICE_SHAPE.default"

    tosa_op = ts.Op.SLICE_SHAPE  # pyrefly: ignore [missing-attribute]
    attr_method = "SliceShapeAttribute"


@register_node_visitor
class TosaModShapeVisitor(TosaBasicShapeVisitor):
    target = "tosa.MOD_SHAPE.default"

    tosa_op = ts.Op.MOD_SHAPE  # pyrefly: ignore [missing-attribute]
    attr_method = "ModShapeAttribute"
