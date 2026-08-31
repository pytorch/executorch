# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

from executorch.backends.arm._passes import ArmOpTargetedPass
from executorch.backends.arm.tosa.dialect.shape import meta_has_shape_mark
from executorch.exir.dialects._ops import ops as exir_ops


class InsertConstShapesPass(ArmOpTargetedPass):
    """Materialize literal shape arguments as CONST_SHAPE nodes.

    This pass targets ops such as `aten.view_copy` and `aten.repeat` whose shape
    arguments might otherwise remain raw Python lists/tuples. Replacing them
    with explicit CONST_SHAPE nodes simplifies the serialization of these ops
    the serialization of their arguments is handled by the CONST_SHAPE node visitor.

    """

    _passes_required_after = set()
    target_ops = {
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.repeat.default,
    }

    def __init__(self) -> None:
        super().__init__()
        self._const_shape_cache: dict[tuple[int, ...], Any] = {}

    @staticmethod
    def _is_shape_arg(arg: Any) -> bool:
        """Return True when `arg` looks like a literal shape list/tuple."""
        is_shape_op = meta_has_shape_mark(arg.meta) if hasattr(arg, "meta") else False
        return (
            not is_shape_op
            and isinstance(arg, (list, tuple))
            and all(isinstance(x, int) for x in arg)
        )

    def call(self, graph_module):
        self._const_shape_cache.clear()
        try:
            return super().call(graph_module)
        finally:
            self._const_shape_cache.clear()

    def call_operator(self, op, args, kwargs, meta, updated: Optional[bool] = False):
        if op not in self.target_ops:
            return super().call_operator(op, args, kwargs, meta, updated)
        if any(InsertConstShapesPass._is_shape_arg(arg) for arg in args):
            new_args = []
            for arg in args:
                if InsertConstShapesPass._is_shape_arg(arg):
                    # Insert a const node for the shape argument
                    if op == exir_ops.edge.aten.view_copy.default:
                        arg = meta.data["val"].shape
                    shape = tuple(arg)
                    const_node = self._const_shape_cache.get(shape)
                    if const_node is None:
                        const_node = super().call_shape_operator(
                            exir_ops.backend.tosa.CONST_SHAPE.default,
                            (arg,),
                            {},
                            meta,
                            True,
                        )
                        self._const_shape_cache[shape] = const_node
                    new_args.append(const_node)
                    updated = True
                else:
                    new_args.append(arg)

            return super().call_operator(op, tuple(new_args), kwargs, meta, updated)

        return super().call_operator(op, args, kwargs, meta, updated)
