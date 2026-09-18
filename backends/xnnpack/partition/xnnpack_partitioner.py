# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import itertools
import logging
from typing import Callable, List, Optional, Type, Union

import torch

from executorch.backends.xnnpack.partition.config import ALL_PARTITIONER_CONFIGS
from executorch.backends.xnnpack.partition.config.xnnpack_config import (
    ConfigPrecisionType,
    XNNPartitionerConfig,
)
from executorch.backends.xnnpack.utils.utils import is_param_node

from executorch.backends.xnnpack.xnnpack_preprocess import XnnpackBackend
from executorch.exir.backend.backend_details import ExportedProgram
from executorch.exir.backend.canonical_partitioners.config_partitioner import (
    ConfigerationBasedPartitioner,
    DSJ,
)
from executorch.exir.backend.partitioner import DelegationSpec
from executorch.exir.passes.constant_prop_pass import (
    constant_prop_pass,
    get_constant_placeholder_dict,
    is_const,
)
from torch.fx.passes.infra.partitioner import Partition

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class XnnpackPartitioner(ConfigerationBasedPartitioner):
    # constant_prop_pass skips aten.full at the edge level so that a scalar
    # fill does not become a stored tensor. Before decomposition the same
    # fills come from these factory ops: the first group decomposes to
    # aten.full, the *_like group to aten.full_like.
    _CONSTANT_PROP_SKIP_TARGETS = frozenset(
        {
            torch.ops.aten.full.default,
            torch.ops.aten.new_full.default,
            torch.ops.aten.ones.default,
            torch.ops.aten.new_ones.default,
            torch.ops.aten.zeros.default,
            torch.ops.aten.new_zeros.default,
            torch.ops.aten.full_like.default,
            torch.ops.aten.ones_like.default,
            torch.ops.aten.zeros_like.default,
        }
    )
    _CONSTANT_PROP_SKIP_NAMESPACES = ("quantized_decomposed", "torchao")

    def __init__(
        self,
        configs: Optional[List[Type[XNNPartitionerConfig]]] = None,
        config_precisions: Optional[
            Union[ConfigPrecisionType, List[ConfigPrecisionType]]
        ] = None,
        per_op_mode=False,
        verbose: bool = False,
        **kwargs,
    ):
        """
        @verbose: if True, print out more information about the partitioner.
            Default level is WARNING. If verbose is True, level is set to DEBUG.
        """
        if verbose:
            logger.setLevel(logging.DEBUG)
            logger.debug("Verbose logging enabled for XNNPACK partitioner.")

        delegation_spec = DelegationSpec(XnnpackBackend.__name__, [])
        configs_to_use = configs or ALL_PARTITIONER_CONFIGS
        # Can do logic and have extra args to filter/delete/select
        # Certain configs based on user specification
        initialized_configs = []
        if isinstance(config_precisions, ConfigPrecisionType):
            config_precisions = [config_precisions]

        for config in configs_to_use:
            # Config Classes given to XnnpackPartitioner should no longer be abstract
            initialized = config(**kwargs)  #  pyre-ignore
            initialized.set_enabled_precision_types(config_precisions)
            initialized_configs.append(initialized)

        # per_op_mode takes the first match from a partitioner config, any
        # subsequent matches that overlap with the first match are not partitioned
        self.per_op_mode = per_op_mode
        super().__init__(delegation_spec, initialized_configs)

    def _check_if_called_from_to_backend(self) -> bool:
        """
        Check if the partition method is being called from the deprecated to_backend workflow.
        Returns True if called from deprecated direct to_backend, False if called from to_edge_transform_and_lower.
        """
        stack = inspect.stack()

        for frame_info in stack:
            if frame_info.function == "to_edge_transform_and_lower":
                return False

        for frame_info in stack:
            if frame_info.function == "to_backend":
                filename = frame_info.filename
                if "program/_program.py" in filename:
                    return True
        return False

    # Pre-decomposition ops that the GEMM configs partition once decomposed,
    # keyed to that config: its weight_idx and bias_idx apply to the op, the
    # conv family takes (input, weight, bias, ...) like aten.convolution and
    # matmul with a 2-d weight lowers to mm.
    _GEMM_TARGETS = {
        torch.ops.aten.linear.default: "linear.default",
        torch.ops.aten.conv1d.default: "convolution.default",
        torch.ops.aten.conv1d.padding: "convolution.default",
        torch.ops.aten.conv2d.default: "convolution.default",
        torch.ops.aten.conv2d.padding: "convolution.default",
        torch.ops.aten.conv_transpose1d.default: "convolution.default",
        torch.ops.aten.conv_transpose2d.input: "convolution.default",
        torch.ops.aten.convolution.default: "convolution.default",
        torch.ops.aten.addmm.default: "addmm.default",
        torch.ops.aten.mm.default: "mm.default",
        torch.ops.aten.matmul.default: "mm.default",
    }

    def transform_for_pre_decomposition(
        self, exported_program: ExportedProgram
    ) -> ExportedProgram:
        """
        Fold the computed weights of the GEMM-like ops into constants.

        The GEMM configs require a static weight and bias, so a convolution
        or a linear whose weight is computed from parameters, for example
        under torch.nn.utils.parametrizations.weight_norm, would otherwise be
        left to the portable kernels together with the weight computation.
        Only those weights are folded: a parameter-only subgraph elsewhere in
        the graph unlocks no delegation and stays an op.
        """
        # A training graph keeps its parameters as inputs: the runtime hands
        # them to the optimizer through the gradient and parameter outputs.
        if exported_program.graph_signature.backward_signature is not None:
            return exported_program
        if not self._computed_gemm_weights(exported_program):
            return exported_program

        # The program is not functionalized yet at this point: a KV-cache
        # update is still an in-place index_put_ or copy_ on a view of the
        # buffer, and the graph signature lists no mutated buffers. Folding a
        # view of a buffer that is written to would leave the write on a
        # constant. Functionalize first; to_edge_transform_and_lower makes
        # the same call right after this hook, and the second call is a
        # no-op on a functional graph.
        exported_program = exported_program.run_decompositions({})

        nodes_to_fold = self._nodes_to_fold(exported_program)
        if not nodes_to_fold:
            return exported_program
        # Buffers stay out of the fold. This hook sees one method at a time,
        # and a buffer this method only reads can be written by another
        # method of the same program.
        return constant_prop_pass(
            exported_program,
            custom_skip_targets=self._constant_prop_skip_targets(exported_program),
            fold_buffers=False,
            nodes_to_fold=nodes_to_fold,
            register_like_source=True,
        )

    def _constant_prop_skip_targets(self, exported_program: ExportedProgram) -> set:
        # Quantization primitives are kept, so that the Q/DQ chain
        # convert_pt2e or torchao's quantize_ leaves on a weight stays in the
        # graph. A folded dequantize would hand the delegate a float weight.
        skip_targets = set(self._CONSTANT_PROP_SKIP_TARGETS)
        for node in exported_program.graph.nodes:
            if (
                node.op == "call_function"
                and getattr(node.target, "namespace", None)
                in self._CONSTANT_PROP_SKIP_NAMESPACES
            ):
                skip_targets.add(node.target)
        return skip_targets

    def _computed_gemm_weights(
        self, exported_program: ExportedProgram
    ) -> List[torch.fx.Node]:
        """
        Returns the weight and bias arguments of the GEMM-like ops that are
        not static, taken from the positions the enabled configs check.
        """
        computed = []
        for node in exported_program.graph.nodes:
            if node.op != "call_function":
                continue
            config = self.target_partitioner_configs.get(
                self._GEMM_TARGETS.get(node.target)
            )
            if config is None or not config.enabled_precision_types:
                continue
            weight_idx, bias_idx = config.weight_idx, config.bias_idx  # pyre-ignore
            if node.target is torch.ops.aten.matmul.default:
                # matmul lowers to mm only with a 2-d weight; bmm takes any.
                weight = node.args[weight_idx]
                if isinstance(weight, torch.fx.Node) and weight.meta["val"].dim() != 2:
                    continue
            for idx in (weight_idx, bias_idx):
                if idx is None or idx >= len(node.args):
                    continue
                arg = node.args[idx]
                if isinstance(arg, torch.fx.Node) and not is_param_node(
                    exported_program, arg
                ):
                    computed.append(arg)
        return computed

    def _constant_only(
        self, exported_program: ExportedProgram
    ) -> Callable[[torch.fx.Node], bool]:
        """
        Returns a predicate for the nodes whose inputs are all parameters,
        lifted constants or such nodes, and that constant_prop_pass would
        fold: not a skipped target, not impure.
        """
        skip_targets = self._constant_prop_skip_targets(exported_program)
        constants: dict = dict.fromkeys(
            get_constant_placeholder_dict(exported_program, fold_buffers=False)
        )
        memo: dict = {}

        def constant_only(node: torch.fx.Node) -> bool:
            if node in memo:
                return memo[node]
            if node.op == "placeholder":
                result = node in constants
            elif (
                node.op != "call_function"
                or node.target in skip_targets
                or node.is_impure()
            ):
                result = False
            else:
                result = all(
                    constant_only(input_node) for input_node in node.all_input_nodes
                ) and (
                    is_const(node.args, exported_program, constants)
                    and is_const(node.kwargs, exported_program, constants)
                )
                if result:
                    constants[node] = None
            memo[node] = result
            return result

        return constant_only

    def _fold_groups(
        self, exported_program: ExportedProgram
    ) -> List[List[torch.fx.Node]]:
        """
        Returns the groups of nodes to fold: each computed weight or bias
        whose inputs are all constant, with the nodes it is computed from
        and the placeholders it reads. Folds that share a node or a source
        are in one group.
        """
        constant_only = self._constant_only(exported_program)
        groups = DSJ()
        for seed in self._computed_gemm_weights(exported_program):
            if not constant_only(seed):
                continue
            stack, seen = [seed], {seed}
            while stack:
                node = stack.pop()
                groups.union(node, seed)
                for input_node in node.all_input_nodes:
                    if input_node not in seen:
                        seen.add(input_node)
                        if input_node.op != "placeholder":
                            stack.append(input_node)
                        else:
                            groups.union(input_node, seed)
        return groups.gen_groups()

    def _nodes_to_fold(self, exported_program: ExportedProgram) -> set[torch.fx.Node]:
        """
        Returns the nodes to fold. A group of folds is applied only if it
        does not make the program larger: the folded tensors may not take
        more bytes than the sources the fold erases. A source used outside
        the fold cannot be erased, so a fold of it would only duplicate it;
        a tied embedding read by one lookup and one transposed matmul is the
        common shape.
        """
        groups = self._fold_groups(exported_program)
        folds = {node for group in groups for node in group if node.op != "placeholder"}
        nodes_to_fold: set[torch.fx.Node] = set()
        for group in groups:
            materialized = [
                node
                for node in group
                if node in folds and any(user not in folds for user in node.users)
            ]
            erased = [
                node
                for node in group
                if node not in folds and all(user in folds for user in node.users)
            ]
            added = sum(self._nbytes(node) for node in materialized)
            removed = sum(self._nbytes(node) for node in erased)
            if added <= removed:
                nodes_to_fold.update(node for node in group if node in folds)
        return nodes_to_fold

    @staticmethod
    def _nbytes(node: torch.fx.Node) -> int:
        val = node.meta["val"]
        return val.numel() * val.element_size()

    def partition(self, exported_program):
        """
        Override partition to add deprecation warning when called from to_backend.
        """
        # Check if we're being called from the deprecated to_backend workflow
        if self._check_if_called_from_to_backend():
            logger.warning(
                "\nDEPRECATION WARNING: You are using the deprecated 'to_edge() + to_backend()' workflow. "
                "Please consider migrating to 'to_edge_transform_and_lower()' for better error handling and optimization. "
            )

        return super().partition(exported_program)

    def generate_partitions(self, ep: ExportedProgram) -> List[Partition]:
        """
        generate_partitions is different if partitioner is set to per_op_mode
        for per_op_mode we only need to generate unmerged partitions instead
        of using the default generate_partitions method.
        """
        if self.per_op_mode:
            return self.generate_per_op_partitions(ep)
        else:
            return super().generate_partitions(ep)

    def generate_per_op_partitions(self, ep: ExportedProgram) -> List[Partition]:
        """
        Uses configs to generate per_op_partitions. That is no partitions are
        merged together. All partitions (node + deps) returned by PartitionerConfigs
        are put into their own partition.
        """
        partitions = []
        matched_nodes = self.get_matched_nodes_from_configs(ep)
        partition_id = itertools.count()
        nodes_seen = {}
        for match in matched_nodes:
            # for debug information we map the node to the string form
            # of the partition it belongs to
            match_map = dict.fromkeys(match, str(match))
            # We only create partitions from the first PartitionerConfig match
            # if a subsequent partitioner match contains the same node, we do
            # not create a partition for it
            overlap = match_map.keys() & nodes_seen.keys()
            if len(overlap) == 0:
                partitions.append(
                    Partition(
                        id=next(partition_id),
                        nodes=match_map.keys(),
                    )
                )
                nodes_seen.update(match_map)
            else:
                error_str = f"per_op mode expects no overlaps between partitions but the partition {match_map.keys()} overlaps with the following partitions:\n"
                for overlap_node in overlap:
                    error_str += f"{nodes_seen[overlap_node]}\n"

                raise RuntimeError(error_str)

        return partitions


class XnnpackDynamicallyQuantizedPartitioner(XnnpackPartitioner):
    def __init__(self, **kwargs):
        if "config_precisions" in kwargs:
            raise ValueError(
                "XnnpackDynamicallyQuantizedPartitioner pins config_precisions to "
                "DYNAMIC_QUANT and does not accept a config_precisions argument."
            )
        super().__init__(
            config_precisions=ConfigPrecisionType.DYNAMIC_QUANT,
            **kwargs,
        )


class XnnpackFloatingPointPartitioner(XnnpackPartitioner):
    def __init__(self):
        super().__init__(config_precisions=ConfigPrecisionType.FP32)


class XnnpackQuantizedPartitioner(XnnpackPartitioner):
    def __init__(self):
        super().__init__(config_precisions=ConfigPrecisionType.STATIC_QUANT)
