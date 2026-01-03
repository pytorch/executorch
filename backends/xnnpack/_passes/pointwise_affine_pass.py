"""
Pointwise Affine Rewrite Pass for XNNPACK.

Converts pointwise Linear/MatMul operations to Conv2d(1x1) or optimized MatMul,
reducing transpose overhead in vision and transformer models.

A pointwise affine operation applies Y = X @ W^T + b independently at each spatial
position. The channel dimension is the contraction dimension for the matrix multiply,
while spatial dimensions are preserved (each spatial location is processed identically).

Algorithm Overview
------------------
MATCHING (PointwiseAffineMatcher):
1. Find Linear/MatMul ops with 2D input [flat_batch, cin]
2. Validate the flatten preserves channel axis:
   - Producer must have exactly ONE axis of size cin
   - That axis must end up at the last position (via permute or already there)
   - Product of other dims must equal flat_batch
3. Trace BACKWARD through layout ops to find N-D origin tensor
4. Find unique channel axis in origin shape (reject if ambiguous)
5. Trace FORWARD through layout ops to find N-D output tensor
   - Require single-user path (avoid breaking shared subgraphs)
   - Accept bias adds only if provably bias (parameter, not activation)
6. Verify output shape matches expected [origin_shape with cout replacing cin]

LOWERING (PointwiseAffineLowering):
- NCHW (rank 4, channel axis 1): Replace entire pattern with Conv2d(1x1)
- Other patterns: Replace with permute -> reshape -> mm -> reshape -> permute

Safety measures to avoid false positives:
1. Unique channel axis - reject if multiple axes could be the channel
2. Single consumer path - require linear has exactly one user
3. Explicit op allowlist - no substring matching
4. Channel axis validation - verify flatten preserves channel intact
5. Bias validation - only accept provably bias adds, not residual connections
6. Concrete shapes only - reject symbolic dimensions

Supported patterns:
- NCHW (rank 4, channel axis 1) -> Conv2d(1x1)
- NHWC/Transformer (channel last) -> MatMul
"""

import copy
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.fx as fx
from executorch.backends.xnnpack._passes.xnnpack_pass import XNNPACKPass
from executorch.backends.xnnpack.utils.utils import get_param_tensor, is_param_node
from executorch.exir.backend.utils import is_shape_dynamic
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export.graph_signature import InputKind
from torch.fx.passes.infra.pass_base import PassResult

logger = logging.getLogger(__name__)


# Op allowlists - explicit sets, no substring matching
LAYOUT_OPS = frozenset(
    [
        torch.ops.aten.permute.default,
        torch.ops.aten.permute_copy.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.view.default,
        torch.ops.aten.view_copy.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.unflatten.int,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.clone.default,
    ]
)
LINEAR_OPS = frozenset(
    [
        torch.ops.aten.linear.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.matmul.default,
    ]
)
RESHAPE_OPS = frozenset(
    [
        torch.ops.aten.view.default,
        torch.ops.aten.view_copy.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
    ]
)
PERMUTE_OPS = frozenset(
    [torch.ops.aten.permute.default, torch.ops.aten.permute_copy.default]
)
TRANSPARENT_OPS = frozenset(
    [torch.ops.aten.contiguous.default, torch.ops.aten.clone.default]
)

# Activation ops that can appear between linear and layout ops.
# We trace through these and recreate them after the rewritten conv2d/mm.
ACTIVATION_OPS = frozenset(
    [
        torch.ops.aten.relu.default,
        torch.ops.aten.gelu.default,
        torch.ops.aten.silu.default,
        torch.ops.aten.hardswish.default,
        torch.ops.aten.hardsigmoid.default,
    ]
)

# Maximum depth for tracing through layout ops (permute, reshape, view, etc.)
# This limits loop iterations when walking the graph to find patterns.
# 20 is generous - real patterns typically have 3-6 layout ops.
_MAX_TRACE_DEPTH = 20


def _underlying(target):
    """Get underlying ATen op for edge dialect ops."""
    return getattr(target, "_op", target)


def _op_in(target, op_set: frozenset) -> bool:
    """Check if target or its underlying op is in the set."""
    return target in op_set or _underlying(target) in op_set


def _shape(node: fx.Node) -> Optional[Tuple[int, ...]]:
    """Get concrete shape from node, None if symbolic."""
    val = node.meta.get("val")
    if val is None or not hasattr(val, "shape"):
        return None
    # Use is_shape_dynamic to efficiently check for symbolic dimensions
    if is_shape_dynamic(node):
        return None
    return tuple(int(s) for s in val.shape)


def _dtype(node: fx.Node) -> torch.dtype:
    """Get dtype from node. Raises if node lacks valid tensor metadata."""
    val = node.meta.get("val")
    if val is None or not hasattr(val, "dtype"):
        raise ValueError(
            f"Node {node.name} lacks valid tensor metadata; cannot extract dtype"
        )
    return val.dtype


def _copy_meta(meta: Dict, new_val=None) -> Dict:
    """
    Copy node metadata, optionally overriding 'val'.
    """
    result = copy.copy(meta)
    if new_val is not None:
        result["val"] = new_val
    return result


class UniqueNameGenerator:
    """
    Generates collision-proof names for constants and buffers.

    Avoids name collisions when:
    - Multiple matches share origin names across scopes
    - Repeated runs of the pass happen
    - Module already has a buffer with that name
    """

    def __init__(self, gm: fx.GraphModule):
        self._existing = set()
        # Collect existing attr names
        for name, _ in gm.named_buffers():
            self._existing.add(name)
        for name, _ in gm.named_parameters():
            self._existing.add(name)
        self._counters: Dict[str, int] = {}

    def __call__(self, prefix: str) -> str:
        """Generate a unique name with the given prefix."""
        # Sanitize prefix
        prefix = prefix.replace(".", "_")

        # Find unique suffix
        if prefix not in self._counters:
            self._counters[prefix] = 0

        while True:
            count = self._counters[prefix]
            self._counters[prefix] += 1
            name = f"{prefix}_{count}" if count > 0 else prefix
            if name not in self._existing:
                self._existing.add(name)
                return name


@dataclass
class Match:
    """Matched pointwise pattern."""

    linear_node: fx.Node
    origin: fx.Node
    output: fx.Node
    weight: torch.Tensor
    bias: Optional[torch.Tensor]
    activation: Optional[fx.Node]  # ReLU, GELU, etc. between linear and layout ops
    channel_axis: int
    cin: int
    cout: int
    shape: Tuple[int, ...]
    nodes: Set[fx.Node]


class PointwiseAffineMatcher:
    """
    Matches pointwise affine patterns where Linear/MatMul operates on flattened
    spatial dimensions with channel preserved as the contraction axis.

    Traces backward from linear to find N-D origin, forward to find N-D output.
    Only matches single-consumer paths to avoid breaking shared subgraphs.
    """

    def __init__(self, gm: fx.GraphModule, ep=None, debug: bool = False):
        self.gm, self.ep, self.debug = gm, ep, debug

    def find_all_matches(self) -> List[Match]:
        """Find all pointwise patterns."""
        return [
            m
            for n in self.gm.graph.nodes
            if n.op == "call_function" and _op_in(n.target, LINEAR_OPS)
            for m in [self._match(n)]
            if m
        ]

    def _match(self, node: fx.Node) -> Optional[Match]:
        """
        Try to match a linear node as pointwise affine.

        Example NCHW pattern being matched:
            origin: [2, 8, 4, 4]  (N=2, C=8, H=4, W=4)
                |
            permute [0,2,3,1] -> [2, 4, 4, 8]
                |
            reshape -> [32, 8]  (flat_batch=32, cin=8)
                |
            linear (weight: [16, 8]) -> [32, 16]  (flat_batch=32, cout=16)
                |
            reshape -> [2, 4, 4, 16]
                |
            permute [0,3,1,2] -> [2, 16, 4, 4]
                |
            output: [2, 16, 4, 4]  (N=2, C=16, H=4, W=4)
        """
        # Validate linear node and extract weight/input info
        info = self._validate_linear_node(node)
        if info is None:
            return None
        weight, w_node, cout, cin, lin_input, flat_batch = info

        # Find origin and channel axis
        origin_info = self._find_origin(lin_input, cin, flat_batch)
        if origin_info is None:
            return None
        origin, fwd_nodes, ch_axis, origin_shape = origin_info

        # Find output, optional bias, and optional activation
        output_info = self._find_output(node, origin_shape, cout)
        if output_info is None:
            return None
        output, bwd_nodes, bias, b_node, add_node, activation = output_info

        # Validate output shape matches expected
        out_shape = _shape(output)
        expected = list(origin_shape)
        expected[ch_axis] = cout
        if out_shape != tuple(expected):
            return None

        # Use bias from linear op if not found in add
        if bias is None:
            bias, b_node = self._get_bias(node, cout)

        nodes = fwd_nodes | bwd_nodes | {node}
        if w_node:
            nodes.add(w_node)
        if b_node:
            nodes.add(b_node)
        if add_node:
            nodes.add(add_node)
        if activation:
            nodes.add(activation)

        return Match(
            node,
            origin,
            output,
            weight,
            bias,
            activation,
            ch_axis,
            cin,
            cout,
            origin_shape,
            nodes,
        )

    def _validate_linear_node(self, node: fx.Node):
        """Validate linear node and return (weight, w_node, cout, cin, lin_input, flat_batch) or None."""
        if len(node.users) != 1:
            return None

        weight, w_node = self._get_weight(node)
        if weight is None or weight.dim() != 2:
            return None
        cout, cin = weight.shape

        lin_input = self._linear_input(node)
        if lin_input is None:
            return None
        in_shape = _shape(lin_input)
        if in_shape is None or len(in_shape) != 2 or in_shape[1] != cin:
            return None
        flat_batch = in_shape[0]

        if not self._valid_flatten(lin_input, cin, flat_batch):
            return None

        return weight, w_node, cout, cin, lin_input, flat_batch

    def _find_origin(self, lin_input: fx.Node, cin: int, flat_batch: int):
        """Find origin tensor and channel axis. Returns (origin, nodes, ch_axis, shape) or None."""
        origin, fwd_nodes = self._trace_back(lin_input)
        if origin is None:
            return None
        origin_shape = _shape(origin)
        if origin_shape is None or len(origin_shape) < 2:
            return None

        ch_axis = self._find_channel_axis(origin_shape, cin, flat_batch)
        if ch_axis is None:
            return None

        return origin, fwd_nodes, ch_axis, origin_shape

    def _find_output(self, node: fx.Node, origin_shape: Tuple[int, ...], cout: int):
        """Find output tensor, bias, and activation.

        Returns (output, nodes, bias, b_node, add_node, activation) or None.
        """
        output, bwd_nodes, add_node, add_pred, activation = self._trace_forward(
            node, len(origin_shape), cout
        )

        bias, b_node = None, None
        if add_node and add_pred:
            add_bias, ab_node = self._extract_add_bias(add_node, add_pred, cout)
            if add_bias is not None:
                bias, b_node = add_bias, ab_node

        if output is None:
            return None

        return output, bwd_nodes, bias, b_node, add_node, activation

    def _find_channel_axis(self, shape, cin, flat_batch) -> Optional[int]:
        """Find unique axis matching cin with correct batch product."""
        # SAFETY: Require exactly ONE candidate. If multiple axes have size==cin and
        # produce the same batch product, we can't determine which is the true channel
        # axis without full axis tracking. Bail rather than guess wrong.
        candidates = [
            i
            for i, s in enumerate(shape)
            if s == cin and _prod_except(shape, i) == flat_batch
        ]
        if len(candidates) == 1:
            # Exactly one axis matches cin and preserves the batch product: treat as channel axis.
            return candidates[0]
        # No candidates: there is no axis with size cin that preserves flat_batch.
        # Multiple candidates: ambiguous channel axis; reject pattern for safety.
        return None

    def _valid_flatten(self, node: fx.Node, cin: int, flat_batch: int) -> bool:
        """
        Validate flatten preserves channel axis intact.

        Ensures cin comes from exactly ONE original axis (not merged from multiple).
        Accepts: flatten.using_ints, reshape/view, or permute that moves cin to last.
        """
        if node.op != "call_function":
            return False

        shape = _shape(node)
        if not shape or len(shape) != 2 or shape[1] != cin:
            return False

        # Check flatten.using_ints
        if _op_in(node.target, {torch.ops.aten.flatten.using_ints}):
            return self._valid_flatten_ints(node, cin)

        # Check reshape/view
        if _op_in(node.target, RESHAPE_OPS):
            return self._valid_reshape(node, cin, flat_batch)

        # Check permute
        if _op_in(node.target, PERMUTE_OPS):
            return self._valid_permute(node, cin, flat_batch)

        # Check transparent ops
        if _op_in(node.target, TRANSPARENT_OPS) and node.args:
            arg = node.args[0]
            if isinstance(arg, fx.Node):
                return self._valid_flatten(arg, cin, flat_batch)

        return False

    def _valid_flatten_ints(self, node: fx.Node, cin: int) -> bool:
        """Validate flatten.using_ints flattens all but last dim."""
        if not node.args or not isinstance(node.args[0], fx.Node):
            return False
        prod_shape = _shape(node.args[0])
        if not prod_shape or len(prod_shape) < 2:
            return False

        start = node.args[1] if len(node.args) > 1 else 0
        end = node.args[2] if len(node.args) > 2 else -1
        rank = len(prod_shape)
        start = start if start >= 0 else rank + start
        end = end if end >= 0 else rank + end

        # Must flatten dims [0, rank-2], keeping last dim intact for valid pointwise.
        # Other flatten ranges could merge cin with spatial dims.
        if start != 0 or end != rank - 2:
            return False

        # cin must be unique and at last position - if multiple axes have size==cin,
        # we can't prove which one survives as the channel dimension.
        cin_axes = [i for i, s in enumerate(prod_shape) if s == cin]
        return len(cin_axes) == 1 and cin_axes[0] == rank - 1

    def _valid_reshape(self, node: fx.Node, cin: int, flat_batch: int) -> bool:
        """Validate reshape is simple flatten-all-but-last."""
        if not node.args or not isinstance(node.args[0], fx.Node):
            return False
        prod = node.args[0]
        prod_shape = _shape(prod)
        if not prod_shape or len(prod_shape) < 2:
            return False

        cin_axes = [i for i, s in enumerate(prod_shape) if s == cin]
        if len(cin_axes) != 1:
            return False

        # cin at last position
        if cin_axes[0] == len(prod_shape) - 1:
            return _prod_except(prod_shape, cin_axes[0]) == flat_batch

        # Check if producer is permute
        if prod.op == "call_function" and _op_in(prod.target, PERMUTE_OPS):
            return self._valid_permute(prod, cin, flat_batch)
        return False

    def _valid_permute(self, node: fx.Node, cin: int, flat_batch: int) -> bool:
        """Validate permute moves cin to last."""
        if len(node.args) < 2:
            return False
        inp, perm = node.args[0], node.args[1]
        if not isinstance(inp, fx.Node) or not isinstance(perm, (list, tuple)):
            return False
        inp_shape = _shape(inp)
        if not inp_shape or len(inp_shape) < 2:
            return False

        cin_axes = [i for i, s in enumerate(inp_shape) if s == cin]
        if len(cin_axes) != 1:
            return False

        perm = list(perm)
        if len(perm) != len(inp_shape):
            return False

        try:
            cin_out = perm.index(cin_axes[0])
        except ValueError:
            return False

        return (
            cin_out == len(perm) - 1
            and _prod_except(inp_shape, cin_axes[0]) == flat_batch
        )

    def _trace_back(self, start: fx.Node) -> Tuple[Optional[fx.Node], Set[fx.Node]]:
        """Trace backward through layout ops."""
        visited, cur = set(), start
        for _ in range(_MAX_TRACE_DEPTH):
            # Cycle detection: check BEFORE processing to fail-fast on first revisit.
            if cur in visited:
                return None, set()
            visited.add(cur)
            if cur.op != "call_function" or not _op_in(cur.target, LAYOUT_OPS):
                return cur, visited
            if cur.args and isinstance(cur.args[0], fx.Node):
                cur = cur.args[0]
            else:
                return None, set()
        else:
            # Loop exhausted without returning - depth limit reached
            logger.debug(
                "PointwiseAffineRewritePass: _trace_back hit depth limit %d at node %s",
                _MAX_TRACE_DEPTH,
                start.name,
            )
        return None, set()

    def _check_target_rank(
        self, cur: fx.Node, target_rank: int, visited, add_node, add_pred, activation
    ):
        """Check if current node is at target rank. Returns result tuple or None to continue."""
        shape = _shape(cur)
        if shape and len(shape) == target_rank:
            return cur, visited, add_node, add_pred, activation
        return None

    def _get_single_user(self, cur: fx.Node) -> Optional[fx.Node]:
        """Get single call_function user of node, or None."""
        users = list(cur.users.keys())
        if len(users) != 1:
            return None
        user = users[0]
        if user.op != "call_function":
            return None
        return user

    def _trace_forward(self, start: fx.Node, target_rank: int, cout: int):
        """
        Trace forward through layout ops to find N-D output.

        Returns (output, visited, add_node, add_predecessor, activation).
        add_predecessor is needed by _extract_add_bias to identify which add arg is bias.
        activation is the single activation op (ReLU, GELU, etc.) if found in the path.

        Canonical activation placement (enforced):
        - linear -> activation -> layout...
        - linear -> bias_add -> activation -> layout...

        After any layout op, we no longer accept activation or bias_add.
        """
        visited, cur = set(), start
        add_node, add_pred = None, None
        activation = None
        seen_layout_op = False

        for _ in range(_MAX_TRACE_DEPTH):
            if cur in visited:
                return None, set(), None, None, None
            visited.add(cur)

            user = self._get_single_user(cur)
            if user is None:
                result = self._check_target_rank(
                    cur, target_rank, visited, add_node, add_pred, activation
                )
                return result if result else (None, set(), None, None, None)

            # Layout ops - continue tracing
            if _op_in(user.target, LAYOUT_OPS):
                seen_layout_op = True
                cur = user
                continue

            # Activation ops - accept at most one, only before layout ops
            if _op_in(user.target, ACTIVATION_OPS):
                if activation is None and not seen_layout_op:
                    activation = user
                    cur = user
                    continue

            # Bias add - only accept before layout ops and before activation
            if _op_in(user.target, {torch.ops.aten.add.Tensor}):
                if not seen_layout_op and activation is None:
                    if self._is_bias_add(user, cur, cout):
                        add_node, add_pred = user, cur
                        cur = user
                        continue

            # Unrecognized op - check if at target rank and stop
            result = self._check_target_rank(
                cur, target_rank, visited, add_node, add_pred, activation
            )
            return result if result else (None, set(), None, None, None)
        else:
            # Loop exhausted without returning - depth limit reached
            logger.debug(
                "PointwiseAffineRewritePass: _trace_forward hit depth limit %d at node %s",
                _MAX_TRACE_DEPTH,
                start.name,
            )
        return None, set(), None, None, None

    def _is_bias_add(self, add_node: fx.Node, cur: fx.Node, cout: int) -> bool:
        """
        Check if add is provably a bias addition (not a residual connection).

        Requires: one input is cur, other is a parameter with valid bias shape.
        """
        if len(add_node.args) < 2:
            return False
        a0, a1 = add_node.args[:2]
        bias_arg = a1 if a0 is cur else (a0 if a1 is cur else None)
        if not isinstance(bias_arg, fx.Node):
            return False
        t = self._get_param(bias_arg)
        return t is not None and self._valid_bias_shape(t.shape, cout)

    def _valid_bias_shape(self, shape, cout: int) -> bool:
        """Check if shape is valid bias (one dim=cout, rest=1)."""
        return (
            len(shape) > 0
            and sum(1 for s in shape if s == cout) == 1
            and all(s in (1, cout) for s in shape)
        )

    def _linear_input(self, node: fx.Node) -> Optional[fx.Node]:
        """Get input to linear op."""
        if _op_in(node.target, {torch.ops.aten.addmm.default}):
            return (
                node.args[1]
                if len(node.args) > 1 and isinstance(node.args[1], fx.Node)
                else None
            )
        return node.args[0] if node.args and isinstance(node.args[0], fx.Node) else None

    def _get_weight(
        self, node: fx.Node
    ) -> Tuple[Optional[torch.Tensor], Optional[fx.Node]]:
        """Get weight tensor."""
        t = node.target
        if _op_in(t, {torch.ops.aten.linear.default}):
            idx = 1
        elif _op_in(t, {torch.ops.aten.addmm.default}):
            idx = 2
        elif _op_in(t, {torch.ops.aten.mm.default, torch.ops.aten.matmul.default}):
            idx = 1
        else:
            return None, None

        if len(node.args) <= idx or not isinstance(node.args[idx], fx.Node):
            return None, None

        w_node = node.args[idx]
        tensor = self._get_param(w_node)
        if tensor is None:
            return None, None

        # Transpose for mm/matmul
        if _op_in(t, {torch.ops.aten.mm.default, torch.ops.aten.matmul.default}):
            tensor = tensor.t().contiguous()
        return tensor, w_node

    def _get_bias(
        self, node: fx.Node, cout: int
    ) -> Tuple[Optional[torch.Tensor], Optional[fx.Node]]:
        """Get bias from linear op."""
        t = node.target
        if (
            _op_in(t, {torch.ops.aten.linear.default})
            and len(node.args) > 2
            and node.args[2]
        ):
            b = node.args[2]
            if isinstance(b, fx.Node):
                tensor = self._get_param(b)
                if tensor is not None and tensor.shape == (cout,):
                    return tensor, b
        elif _op_in(t, {torch.ops.aten.addmm.default}) and node.args:
            b = node.args[0]
            if isinstance(b, fx.Node):
                tensor = self._get_param(b)
                if tensor is not None and tensor.shape == (cout,):
                    return tensor, b
        return None, None

    def _extract_add_bias(self, add_node: fx.Node, pred: fx.Node, cout: int):
        """
        Extract bias tensor from separate add node.

        Uses pred to identify which arg is the activation (not bias). This prevents
        picking the wrong arg if both happen to have compatible shapes.
        """
        if len(add_node.args) < 2:
            return None, None
        a0, a1 = add_node.args[:2]
        bias_arg = a1 if a0 is pred else (a0 if a1 is pred else None)
        if not isinstance(bias_arg, fx.Node):
            return None, None
        t = self._get_param(bias_arg)
        if t is not None and self._valid_bias_shape(t.shape, cout):
            return t.reshape(cout), bias_arg
        return None, None

    def _get_param(self, node: fx.Node) -> Optional[torch.Tensor]:
        """
        Get parameter tensor from node. Returns None for activations.

        Only returns tensors that are provably parameters/constants to avoid
        treating computed tensors (reduces, gathers) as bias.
        NOTE: Does NOT use node.meta["val"] as that includes activations.

        Uses get_param_tensor and is_param_node from utils when ep is available,
        falls back to get_attr for non-edge graphs.
        """
        # Use utilities from utils.py when ep is available
        if self.ep:
            if is_param_node(self.ep, node):
                return get_param_tensor(self.ep, node)
            return None

        # Fallback for non-edge graphs: only support get_attr nodes
        if node.op == "get_attr":
            try:
                return getattr(self.gm, node.target)
            except AttributeError:
                pass
        return None


def _prod_except(shape, exclude_idx: int) -> int:
    """
    Product of shape excluding one index.

    Args:
        shape: Tuple or list of dimension sizes. Must be non-empty.
        exclude_idx: Index to exclude. Must be in range [0, len(shape)).

    Returns:
        Product of all dimensions except the one at exclude_idx.

    Raises:
        IndexError: If exclude_idx is out of bounds.
        ValueError: If shape is empty.
    """
    if not shape:
        raise ValueError("_prod_except requires non-empty shape")
    if exclude_idx < 0 or exclude_idx >= len(shape):
        raise IndexError(
            f"exclude_idx {exclude_idx} out of range for shape with {len(shape)} dims"
        )
    p = 1
    for i, s in enumerate(shape):
        if i != exclude_idx:
            p *= s
    return p


class PointwiseAffineLowering:
    """
    Lowers matched pointwise patterns to optimized operations.

    Strategy:
    - NCHW (rank 4, channel axis 1): Conv2d(1x1) - eliminates all transposes
    - All other patterns: MatMul with explicit permute/reshape

    INVARIANT: We NEVER mutate or reshape existing parameters/buffers in-place.
    We always materialize NEW constants for the rewritten subgraph. This is
    critical when weights are shared across multiple linear nodes (e.g., tied
    embeddings in LLMs).
    """

    def __init__(self, gm: fx.GraphModule, ep=None):
        self.gm, self.ep = gm, ep
        self._name_gen = UniqueNameGenerator(gm)

    def lower(self, m: Match) -> fx.Node:
        """Lower match to optimized ops."""
        if len(m.shape) == 4 and m.channel_axis == 1:
            return self._to_conv2d(m)
        return self._to_matmul(m)

    def _create_node(
        self,
        graph: fx.Graph,
        target: Callable,
        args: Tuple,
        meta_like: fx.Node,
        new_val: torch.Tensor,
    ) -> fx.Node:
        """
        Create a node with properly copied metadata.
        """
        node = graph.call_function(target, args)
        node.meta = _copy_meta(meta_like.meta, new_val)
        return node

    def _meta_tensor(self, shape: List[int], dtype: torch.dtype) -> torch.Tensor:
        """
        Create a tensor for metadata purposes only.

        Uses torch.empty to avoid unnecessary memory initialization since only
        shape and dtype are used, not the tensor values.
        """
        return torch.empty(shape, dtype=dtype)

    def _create_activation(
        self,
        graph: fx.Graph,
        activation: fx.Node,
        inp: fx.Node,
        out_shape: List[int],
        dtype: torch.dtype,
    ) -> fx.Node:
        """
        Recreate activation op (ReLU, GELU, etc.) with new input.

        Preserves:
        - Extra arguments from the original activation node
          (e.g., negative_slope for leaky_relu, approximate for gelu)
        - Metadata from the original activation node (qparams, ranges, etc.)
          with updated 'val' for the new output shape
        """
        # Preserve extra args beyond the input tensor
        extra_args = activation.args[1:] if len(activation.args) > 1 else ()
        # Use activation as meta_like to preserve its metadata (qparams, ranges, etc.)
        return self._create_node(
            graph,
            activation.target,
            (inp,) + extra_args,
            activation,  # Copy metadata from original activation, not inp
            self._meta_tensor(out_shape, dtype),
        )

    def _to_conv2d(self, m: Match) -> fx.Node:
        """Lower to Conv2d(1x1)."""
        g = self.gm.graph
        dtype = _dtype(m.origin)
        # INVARIANT: Create NEW weight tensor, never mutate original
        w = m.weight.reshape(m.cout, m.cin, 1, 1)

        with g.inserting_before(m.output):
            w_node = self._const(w, f"{m.origin.name}_conv_w")
            b_node = (
                self._const(m.bias, f"{m.origin.name}_conv_b")
                if m.bias is not None
                else None
            )

            out_shape = list(m.shape)
            out_shape[m.channel_axis] = m.cout

            # Use edge op for compatibility with FuseActivationPass
            out = self._create_node(
                g,
                exir_ops.edge.aten.convolution.default,
                (m.origin, w_node, b_node, [1, 1], [0, 0], [1, 1], False, [0, 0], 1),
                m.origin,
                self._meta_tensor(out_shape, dtype),
            )

            # Insert activation if present (e.g., ReLU, GELU)
            if m.activation is not None:
                out = self._create_activation(g, m.activation, out, out_shape, dtype)

        return out

    def _to_matmul(self, m: Match) -> fx.Node:
        """Lower to MatMul."""
        g = self.gm.graph
        dtype = _dtype(m.origin)
        # INVARIANT: Create NEW weight tensor, never mutate original
        w = m.weight.t().contiguous()
        rank = len(m.shape)
        ch = m.channel_axis

        with g.inserting_before(m.output):
            w_node = self._const(w, f"{m.origin.name}_mm_w")
            inp = m.origin

            # Permute channel to last if needed
            need_perm = ch != rank - 1
            if need_perm:
                perm = [i for i in range(rank) if i != ch] + [ch]
                perm_shape = [m.shape[i] for i in perm]
                inp = self._create_node(
                    g,
                    torch.ops.aten.permute.default,
                    (inp, perm),
                    m.origin,
                    self._meta_tensor(perm_shape, dtype),
                )

            # Flatten
            flat = _prod_except(m.shape, ch)
            flat_node = self._create_node(
                g,
                torch.ops.aten.reshape.default,
                (inp, [flat, m.cin]),
                m.origin,
                self._meta_tensor([flat, m.cin], dtype),
            )

            # MatMul - use edge op for compatibility with downstream passes
            out = self._create_node(
                g,
                exir_ops.edge.aten.mm.default,
                (flat_node, w_node),
                m.origin,
                self._meta_tensor([flat, m.cout], dtype),
            )

            # Bias - use edge op for compatibility with FuseActivationPass
            if m.bias is not None:
                b_node = self._const(m.bias.view(1, m.cout), f"{m.origin.name}_mm_b")
                out = self._create_node(
                    g,
                    exir_ops.edge.aten.add.Tensor,
                    (out, b_node),
                    m.origin,
                    self._meta_tensor([flat, m.cout], dtype),
                )

            # Unflatten
            nhwc = [s for i, s in enumerate(m.shape) if i != ch] + [m.cout]
            out = self._create_node(
                g,
                torch.ops.aten.reshape.default,
                (out, nhwc),
                m.origin,
                self._meta_tensor(nhwc, dtype),
            )

            # Permute back
            if need_perm:
                perm_back = list(range(rank - 1))
                perm_back.insert(ch, rank - 1)
                final = list(m.shape)
                final[ch] = m.cout
                out = self._create_node(
                    g,
                    torch.ops.aten.permute.default,
                    (out, perm_back),
                    m.origin,
                    self._meta_tensor(final, dtype),
                )

            # Insert activation if present (e.g., ReLU, GELU)
            if m.activation is not None:
                out_shape = list(m.shape)
                out_shape[ch] = m.cout
                out = self._create_activation(g, m.activation, out, out_shape, dtype)

        return out

    def _const(self, tensor: Optional[torch.Tensor], prefix: str) -> Optional[fx.Node]:
        """
        Create constant node with collision-proof naming.

        For EdgeProgramManager, uses create_constant_placeholder which requires
        constants to be inserted before user input nodes. Falls back to get_attr
        for non-edge graphs or when no user input is found.
        """
        if tensor is None:
            return None

        name = self._name_gen(prefix)

        if self.ep:
            from executorch.backends.transforms.utils import create_constant_placeholder

            # Find the first user input placeholder to insert before
            graph = self.gm.graph
            first_user_input = None
            for node in graph.nodes:
                if node.op == "placeholder":
                    # Check if this is a user input (not a parameter/buffer)
                    sig = self.ep.graph_signature
                    if node.name in [
                        spec.arg.name
                        for spec in sig.input_specs
                        if spec.kind == InputKind.USER_INPUT
                    ]:
                        first_user_input = node
                        break

            if first_user_input:
                with graph.inserting_before(first_user_input):
                    return create_constant_placeholder(
                        self.ep, graph, name, InputKind.PARAMETER, tensor
                    )
            else:
                # No user input found - unexpected for a valid edge program.
                # Fall through to get_attr fallback.
                logger.debug(
                    "PointwiseAffineRewritePass: no user input found in graph, "
                    "using get_attr fallback for constant %s",
                    name,
                )

        # Fallback: use get_attr (works for non-edge graphs)
        self.gm.register_buffer(name, tensor)
        node = self.gm.graph.get_attr(name)
        node.meta["val"] = tensor
        return node


class PointwiseAffineRewritePass(XNNPACKPass):
    """
    Pass that rewrites pointwise Linear/MatMul to Conv2d(1x1) or optimized MatMul.

    Safe by design: rejects ambiguous patterns rather than risk miscompilation.
    Use via to_edge_transform_and_lower(program, transform_passes=[PointwiseAffineRewritePass()]).
    """

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        ep = self.exported_program
        matcher = PointwiseAffineMatcher(graph_module, ep)
        matches = matcher.find_all_matches()

        if not matches:
            return PassResult(graph_module, False)

        lowering = PointwiseAffineLowering(graph_module, ep)
        for m in matches:
            new_out = lowering.lower(m)
            m.output.replace_all_uses_with(new_out)

            # Erase unused nodes in reverse topological order (users before inputs).
            # Sort match nodes by graph position, then reverse to get proper erase order.
            nodes_to_erase = [
                node
                for node in graph_module.graph.nodes
                if node in m.nodes and node != m.origin
            ]
            for node in reversed(nodes_to_erase):
                if len(node.users) == 0:
                    # For placeholder nodes (parameters/buffers), use proper cleanup
                    if node.op == "placeholder" and ep:
                        from executorch.backends.transforms.utils import (
                            delete_constant_placeholder,
                        )

                        delete_constant_placeholder(ep, node)
                        continue
                    graph_module.graph.erase_node(node)

        graph_module.graph.lint()
        graph_module.recompile()
        return PassResult(graph_module, True)
