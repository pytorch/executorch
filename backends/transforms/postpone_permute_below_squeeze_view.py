# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import copy
from typing import cast, List

import torch
import torch.fx
from executorch.backends.transforms.permute_pass_utils import (
    get_shape,
    RemoveOrReplacePassInterface,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import PassResult


class PostponePermuteOpBelowSqueezeOrUnsqueezeLikeView(RemoveOrReplacePassInterface):
    """
    A common pattern seen in transformer models.  If the consumer of permute
    is a view op, swap their order so permute is below view.
    Change "permute -> view" to "view -> permute"
    This is to optimize a chain of view->permute->view->permute...
    so that the chain will be become view->v...->view->permute->p...->permute.
    The chain can be optimized by FuseCascadedTransposeOrPermuteOps() and
    FuseCascadedViewOps().
    Notice the class name has ViewSqueeze to indicate the View is
    functionally the same as a squeeze or unsqueeze. It does not necessarily
    mean the view_copy is normalized from squeeze or unsqueeze.
    """

    @property
    def targets(self) -> list[EdgeOpOverload]:
        return [exir_ops.edge.aten.permute_copy.default]

    # If list1 and list2 are same (same values and in same order) except
    # list1 has one more element with value of 1. Return index of the extra 1.
    # Otherwise return -1.
    def check_if_shapes_differ_in_single_dim_of_size_1(
        self, list1: List, list2: List
    ) -> int:
        if len(list1) != len(list2) + 1:
            return -1
        for i in range(len(list2)):
            if list1[i] != list2[i]:
                # Return index of the extra 1 if the remaining parts are the same
                if list1[i] == 1 and list2[i:] == list1[i + 1 :]:
                    return i
                else:
                    return -1
        # If no difference was found, the extra element is at the end
        if list1[-1] == 1:
            return len(list2)
        else:
            return -1

    def maybe_remove_or_replace(self, node: torch.fx.Node) -> bool:
        users = list(node.users.keys())
        # Transform only for pattern permute_copy->view_copy, and
        # view_copy op is the only user of permute_copy.
        if len(users) != 1 or users[0].target not in (
            exir_ops.edge.aten.view_copy.default,
            exir_ops.edge.aten.view.default,
        ):
            return False

        # If the permute_node/view_node was newly added to the
        # graph, it may not have the meta["val"] FakeTensor.
        # Skip in this case.
        if node.meta.get("val") is None:
            return False

        permute_node_shape = [*cast(list, get_shape(node.graph.owning_module, node))]

        permute_dims = cast(list, node.args[1])
        view_node = users[0]

        if view_node.meta.get("val") is None:
            return False

        view_node_shape = [*cast(list, get_shape(node.graph.owning_module, view_node))]

        pred = node.args[0]
        if not isinstance(pred, torch.fx.Node) or pred.meta.get("val") is None:
            return False

        pred_shape = [*cast(list, get_shape(node.graph.owning_module, pred))]

        # Handle three cases
        # 1. view_node_shape is almost same as permute_node_shape
        #    except the view_node has one more dim somewhere
        #    and the extra dim has value of 1.
        # 2. view_node_shape is almost same as permute_node_shape
        #    except permute_node_shape has one more dim somewhere
        #    and the extra dim has value of 1.
        # 3. view_node_shape is the same as permute_node_shape.

        if len(permute_node_shape) + 1 == len(view_node_shape):
            index = self.check_if_shapes_differ_in_single_dim_of_size_1(
                view_node_shape, permute_node_shape
            )
            if index != -1:
                # view_node_shape is almost same as permute_node_shape
                # except it has one more dim somewhere
                # and the extra dim has value of 1.
                new_view_shape = copy.deepcopy(pred_shape)
                new_view_shape.insert(index, 1)
                new_permute_dims = [x + 1 if x >= index else x for x in permute_dims]
                new_permute_dims.insert(index, index)
                self._insert_nodes(
                    node.graph,
                    pred,
                    node,
                    view_node,
                    new_view_shape,
                    new_permute_dims,
                )
                return True

        elif len(view_node_shape) + 1 == len(permute_node_shape):
            index = self.check_if_shapes_differ_in_single_dim_of_size_1(
                permute_node_shape, view_node_shape
            )
            if index != -1:
                # view_node_shape is almost same as permute_node_shape
                # except permute_node_shape has one more dim somewhere
                # and the extra dim has value of 1.
                # Convert permute_dims to list of ints
                index_to_remove = permute_dims[index]
                new_view_shape = copy.deepcopy(pred_shape)
                del new_view_shape[index_to_remove]
                new_permute_dims = [
                    x - 1 if x > index_to_remove else x for x in permute_dims
                ]
                del new_permute_dims[index]
                self._insert_nodes(
                    node.graph,
                    pred,
                    node,
                    view_node,
                    new_view_shape,
                    new_permute_dims,
                )
                return True

        elif permute_node_shape == view_node_shape:
            # view_node_shape is the same as permute_node_shape
            # Replace the uses of view_node with permute_node
            view_node.replace_all_uses_with(node)
            return True

        return False

    def _insert_nodes(
        self,
        graph: torch.fx.Graph,
        pred: torch.fx.Node,
        permute_node: torch.fx.Node,
        view_node: torch.fx.Node,
        new_view_shape: List,
        new_permute_dims: List,
    ) -> None:
        with graph.inserting_after(view_node):
            # Target is guaranteed to be a callable since it's from the graph
            view_target = view_node.target
            assert callable(view_target), "View target must be callable"
            new_view_node = graph.call_function(
                view_target,
                args=(pred, new_view_shape),
            )

        with graph.inserting_after(new_view_node):
            # Target is guaranteed to be a callable since it's from our targets list
            permute_target = permute_node.target
            assert callable(permute_target), "Permute target must be callable"
            new_permute_node = graph.call_function(
                permute_target,
                args=(new_view_node, new_permute_dims),
            )
            new_permute_node.meta = view_node.meta
            view_node.replace_all_uses_with(new_permute_node)

        # view_node is user of permute_node, so must erase view_node first
        graph.erase_node(view_node)
        graph.erase_node(permute_node)

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        # This pass needs to iterate until convergence because postponing
        # one permute may enable postponing another in a chain
        iter_count = 0
        local_modified = False
        overall_modified = False
        while local_modified or iter_count == 0:
            result = super().call(graph_module)
            local_modified = result.modified
            overall_modified |= local_modified
            graph_module = result.graph_module
            iter_count += 1
            if iter_count == 4:
                break

        return PassResult(graph_module, overall_modified)
