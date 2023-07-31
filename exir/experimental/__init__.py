# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from typing import List, Optional, Tuple, Union

import torch
import torch.utils._pytree as pytree
from executorch.exir.tensor import TensorSpec
from torch._export.serde.schema import TensorMeta
from torch._export.serde.serialize import (
    _SERIALIZE_TO_TORCH_DTYPE,
    serialize_tensor_meta,
)
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv


def add_assertions(graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    modified_graph_module = copy.deepcopy(graph_module)

    graph = modified_graph_module.graph
    for node in graph.nodes:
        if node.op != "call_function" and node.op != "placeholder":
            continue

        # Ignore constants
        if node.meta.get("val", None) is None:
            continue

        # Ignore non-torch ops
        if node.op == "call_function" and (
            not isinstance(node.target, torch._ops.OpOverload)
        ):
            continue

        shape = node.meta["val"].shape
        dtype = node.meta["val"].dtype
        node_name = node.name
        with graph.inserting_after(node):

            def check_spec(
                x: TensorSpec, shape: List[int], dtype: torch.dtype, node_name: str
            ) -> None:
                assert list(x.shape) == list(
                    shape
                ), f"Expected {node_name} shape to be {shape}, got {x.shape}"
                assert (
                    x.dtype == dtype
                ), f"Expected {node_name} dtype to be {dtype}, got {x.dtype}"

            graph.call_function(check_spec, (node, shape, dtype, node_name))

    modified_graph_module.recompile()
    return modified_graph_module


def convert_fake_tensor_to_tensor_meta(
    ep: torch.fx.GraphModule,
) -> Tuple[torch.fx.GraphModule, Optional[ShapeEnv]]:
    """
    Replace the faketensor metadata with the tensor metadata dataclass since we
    cannot serialize faketensors
    """
    shape_env = None
    for node in ep.graph.nodes:

        def get_shape_env(
            val: Union[List[FakeTensor], FakeTensor]
        ) -> Optional[ShapeEnv]:
            val_flat, _ = pytree.tree_flatten(val)
            curr_shape_env = None
            for v in val_flat:
                if not isinstance(v, FakeTensor):
                    continue
                if curr_shape_env is None:
                    curr_shape_env = v.fake_mode.shape_env
                else:
                    assert (
                        curr_shape_env is v.fake_mode.shape_env
                    ), "Multiple shape envs detected."
            return curr_shape_env

        if (val := node.meta.get("val", None)) is not None:
            if shape_env is None:
                shape_env = get_shape_env(val)
            elif (new_shape_env := get_shape_env(val)) is not None:
                assert shape_env is new_shape_env, "Multiple shape envs detected."

            node.meta["tensor_meta"] = pytree.tree_map_only(
                torch.Tensor, serialize_tensor_meta, val
            )
            del node.meta["val"]

    return ep, shape_env


def convert_tensor_meta_to_fake_tensor(
    ep: torch.fx.GraphModule, shape_env: Optional[ShapeEnv] = None
) -> torch.fx.GraphModule:
    """
    Replace (inplace) the tensor metadata with faketensor
    """
    fake_tensor_mode: FakeTensorMode = FakeTensorMode(
        allow_non_fake_inputs=True, shape_env=shape_env
    )
    for node in ep.graph.nodes:
        if (val := node.meta.get("tensor_meta", None)) is not None:

            def _extract_faketensor(tensor_meta: TensorMeta) -> FakeTensor:
                return FakeTensor(
                    fake_tensor_mode,
                    torch.empty(
                        # TODO Support dynamic shape.
                        tuple(s.as_int for s in tensor_meta.sizes),
                        dtype=_SERIALIZE_TO_TORCH_DTYPE[tensor_meta.dtype],
                        device="meta",
                        requires_grad=tensor_meta.requires_grad,
                    ),
                    torch.device("cpu"),
                )

            node.meta["val"] = pytree.tree_map_only(
                TensorMeta, _extract_faketensor, val
            )
    return ep
