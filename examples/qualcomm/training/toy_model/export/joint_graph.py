# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Joint (forward+backward) graph creation for QNN HTP training.

Trimmed for a bias-free toy model + MSE loss: only the one decomposition such a
graph actually needs is registered (full-tensor ``mean``); QNN-native ops are
left undecomposed via ``get_skip_decomp_table()``.
"""

import logging
from contextlib import ExitStack

import torch

from executorch.backends.qualcomm.partition.qnn_partitioner import get_skip_decomp_table
from torch import nn
from torch._decomp import core_aten_decompositions, remove_decompositions
from torch._functorch.aot_autograd import aot_export_joint_with_descriptors

# _PyTreeCodeGen/_PyTreeInfo are private torch internals; verified against
# PyTorch 2.13. Graph.set_codegen() (public) is used for the write path;
# ._codegen is still needed to read orig_args/in_spec (no public getter exists).
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.utils._pytree import tree_flatten

logger = logging.getLogger(__name__)


class JointGraphCreator:
    """Builds the forward+backward joint graph via the AOT descriptor API.

    The returned ``GraphModule`` has no ``nn.Parameter`` attributes -- every
    weight is an anonymous ``primals_N`` placeholder input. That is what lets
    a later ``torch.export.export`` re-trace of this graph (done internally
    by ``to_edge_transform_and_lower_to_qnn``) classify every weight as a
    plain ``USER_INPUT``, with no manual ``ExportedProgram`` construction.
    """

    def __init__(self, module: nn.Module, forward_inputs: tuple, decompositions=None):
        self.module = module
        self.forward_inputs = forward_inputs
        self.decompositions = decompositions or self.build_decompositions()
        self._joint_desc = None
        self._gm = None

    @staticmethod
    def build_decompositions() -> dict:
        decomps = core_aten_decompositions()

        # QNN-native ops stay undecomposed.
        remove_decompositions(decomps, get_skip_decomp_table())

        # mse_loss's reduction="mean" produces a full-tensor mean; QNN has no
        # mean.default builder, only mean.dim.
        decomps[torch.ops.aten.mean.default] = _mean_default_decomp
        return decomps

    def create(self) -> torch.fx.GraphModule:
        with ExitStack() as stack:
            self._joint_desc = aot_export_joint_with_descriptors(
                stack,
                self.module,
                self.forward_inputs,
                decompositions=self.decompositions,
            )
            gm = self._joint_desc.graph_module

        gm = self._flatten_outputs_and_remove_none(gm)
        gm = self._remove_unused_placeholders(gm)
        self._gm = gm
        return gm

    @property
    def joint_desc(self):
        return self._joint_desc

    @property
    def graph_module(self):
        return self._gm

    def _remove_unused_placeholders(self, gm):
        """Remove unused parameter/buffer placeholders.

        In the joint graph, aot_export_joint assigns primals_N to all params
        then all buffers in sequence. User inputs (forward primals beyond
        that range and tangents) must never be removed.
        """
        params_spec = list(self._joint_desc.params_spec or [])
        buffers_spec = list(self._joint_desc.buffers_spec or [])
        n_params = len(params_spec)
        n_buffers = len(buffers_spec)

        def _is_user_input(name):
            if name.startswith("tangents_"):
                return True
            if name.startswith("primals_"):
                try:
                    return int(name[len("primals_") :]) - 1 >= n_params + n_buffers
                except ValueError:
                    pass
            return False

        changed = True
        while changed:
            changed = False
            for node in list(gm.graph.nodes):
                if (
                    node.op == "placeholder"
                    and not _is_user_input(node.name)
                    and len(node.users) == 0
                ):
                    logger.info("Removing unused placeholder: %s", node.name)
                    gm.graph.erase_node(node)
                    changed = True
        gm.graph.lint()
        gm.recompile()
        return gm

    def _flatten_outputs_and_remove_none(self, gm):
        """Flatten output node, remove None entries, update meta['val'], and
        reset the codegen pytree out_spec so torch.fx.Interpreter can process
        the outputs without a leaf-count mismatch.
        """
        new_outputs = []
        for node in gm.graph.nodes:
            if node.op == "output":
                flat_outputs, _ = tree_flatten(node.args[0])
                new_outputs = [o for o in flat_outputs if o is not None]
                node.args = (new_outputs,)
                if "val" in node.meta and isinstance(node.meta["val"], (tuple, list)):
                    node.meta["val"] = tuple(
                        v for v in node.meta["val"] if v is not None
                    )
                break

        gm.graph.lint()

        if new_outputs:
            codegen = gm.graph._codegen  # pylint: disable=protected-access
            if isinstance(codegen, _PyTreeCodeGen) and codegen.pytree_info is not None:
                dummy_out = tuple(torch.empty(1) for _ in new_outputs)
                _, new_out_spec = tree_flatten(dummy_out)
                try:
                    old_info = codegen.pytree_info
                    gm.graph.set_codegen(
                        _PyTreeCodeGen(
                            _PyTreeInfo(
                                old_info.orig_args,
                                old_info.in_spec,
                                new_out_spec,
                            )
                        )
                    )
                except (AttributeError, TypeError) as exc:
                    logger.warning(
                        "Could not reset pytree out_spec (%s); export will "
                        "likely fail with a leaf-count mismatch.",
                        exc,
                    )

        gm.recompile()
        return gm


def _mean_default_decomp(self):
    """mean.default (full-tensor mean) -> mean.dim, which QNN supports."""
    dims = list(range(self.dim()))
    return torch.ops.aten.mean.dim(self, dims)
