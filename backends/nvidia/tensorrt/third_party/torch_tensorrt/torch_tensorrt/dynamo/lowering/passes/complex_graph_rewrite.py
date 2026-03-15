import logging
from typing import Callable, List, Set, Tuple

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import GraphModule, Node
from torch.fx.experimental.proxy_tensor import unset_fake_temporarily
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


class ComplexSubGraphInfo:
    def __init__(
        self,
        anchor_nodes: List[Node],
        subgraph_nodes: List[Node],
        input_nodes: List[Node],
    ):
        self.anchor_nodes = anchor_nodes
        self.subgraph_nodes = subgraph_nodes
        self.input_nodes = input_nodes

    def __repr__(self) -> str:
        return (
            f"ComplexOpSubGraphInfo(anchor_nodes={[n.name for n in self.anchor_nodes]}, "
            f"subgraph={[n.name for n in self.subgraph_nodes]}, "
            f"inputs={[n.name for n in self.input_nodes]})"
        )


class ComplexOpDetector:
    def __init__(self) -> None:
        pass

    def is_complex_dtype(self, node: Node) -> bool:
        # Check if node's metadata or dtype is complex
        dtype = None
        if "val" in node.meta:
            val = node.meta["val"]
            if hasattr(val, "dtype"):
                dtype = val.dtype

        logger.debug(f"dtype of node: {dtype}")
        return dtype in {torch.complex64, torch.complex128}

    def node_include_in_subgraph(self, node: Node) -> bool:
        # Include only call_function ops on complex tensors
        if node.op == "call_function" and self.is_complex_dtype(node):
            logger.debug(
                f"node.op is added to subgraph: {node.op}, node name: {node.name} is complex"
            )
        return node.op == "call_function" and self.is_complex_dtype(node)

    def subgraph_from_anchor(self, anchor_node: Node) -> ComplexSubGraphInfo:
        subgraph_nodes: Set[Node] = set()
        input_nodes: Set[Node] = set()
        stack = [anchor_node]
        while stack:
            n = stack.pop()
            if n in subgraph_nodes:
                continue
            subgraph_nodes.add(n)
            logger.debug(f"node {n.name} is added to subgraph")
            for inp in n.all_input_nodes:
                if self.node_include_in_subgraph(inp):
                    stack.append(inp)
                else:
                    input_nodes.add(inp)
        return ComplexSubGraphInfo(
            [anchor_node], list(subgraph_nodes), list(input_nodes)
        )

    def find_complex_op_subgraphs(
        self, gm: GraphModule, anchor_target: str
    ) -> List[ComplexSubGraphInfo]:
        complex_op_subgraphs: List[ComplexSubGraphInfo] = []
        for node in gm.graph.nodes:
            if node.target == anchor_target:
                new_sub = self.subgraph_from_anchor(node)
                # if any intersecting nodes between seen and sub.subgraph_nodes they should be merged
                merged = False
                for existing_sub in complex_op_subgraphs:
                    if set(existing_sub.subgraph_nodes) & set(new_sub.subgraph_nodes):
                        logger.debug(f"merging subgraphs {existing_sub} {new_sub}")
                        # merge the two subgraphs
                        existing_sub.subgraph_nodes = list(
                            set(existing_sub.subgraph_nodes)
                            | set(new_sub.subgraph_nodes)
                        )
                        existing_sub.input_nodes = list(
                            set(existing_sub.input_nodes) | set(new_sub.input_nodes)
                        )
                        existing_sub.anchor_nodes = list(
                            set(existing_sub.anchor_nodes) | set(new_sub.anchor_nodes)
                        )
                        merged = True
                        break
                if not merged:
                    complex_op_subgraphs.append(new_sub)
        return complex_op_subgraphs


class ComplexGraphRewriter:
    def __init__(self, gm: GraphModule, truncate_double: bool = False) -> None:
        self.gm = gm
        self.truncate_double = truncate_double

    def extract_shape_dtype_device(
        self, input_node: Node
    ) -> Tuple[Tuple[int, ...], torch.dtype, torch.device]:
        if input_node.op == "placeholder":
            tensor_val = input_node.meta["val"]

        elif input_node.op == "get_attr":
            tensor_val = self.get_attr_tensor(input_node.target)  # type: ignore

        else:
            raise ValueError(f"Unsupported node type: {input_node.op}")

        node_shape = tensor_val.size()
        dtype = tensor_val.dtype
        new_node_shape = node_shape + (2,)
        device = tensor_val.device

        if dtype == torch.complex64:
            new_node_dtype = torch.float32
        elif dtype == torch.complex128 and self.truncate_double:
            new_node_dtype = torch.float32
        else:
            new_node_dtype = torch.float64

        return new_node_shape, new_node_dtype, device

    def get_attr_tensor(self, target):  # type: ignore
        # Check if target is param or buffer
        if target in dict(self.gm.named_parameters()):
            return self.gm.get_parameter(target)
        elif target in dict(self.gm.named_buffers()):
            return self.gm.get_buffer(target)
        else:
            raise ValueError(
                f"Attribute {target} not found in gm parameters or buffers."
            )

    def replace_input_node(self, input_node: Node) -> None:
        modified = False
        logger.debug(f"Replacing input node: {input_node.name}")
        new_shape, new_dtype, device = self.extract_shape_dtype_device(input_node)
        real_tensor = torch.empty(new_shape, dtype=new_dtype, device=device)

        if input_node.op == "placeholder":
            with FakeTensorMode() as fake_mode:
                fake_tensor = fake_mode.from_tensor(real_tensor)
            with self.gm.graph.inserting_before(input_node):
                new_node = self.gm.graph.placeholder(input_node.target + "_reshaped")
            new_node.meta["val"] = fake_tensor

        elif input_node.op == "get_attr":
            new_attr_name = input_node.target + "_reshaped"
            with unset_fake_temporarily():
                original_tensor = self.get_attr_tensor(input_node.target)  # type: ignore
                stacked_tensor = torch.stack(
                    [original_tensor.real, original_tensor.imag], dim=-1
                )
                self.gm.register_buffer(new_attr_name, stacked_tensor)
            with self.gm.graph.inserting_after(input_node):
                new_node = self.gm.graph.get_attr(new_attr_name)
        else:
            logger.debug(
                f"Unsupported node type in replacement of input node: {input_node.op}"
            )
            logger.debug(
                "This complex subgraph inputnode type does not need to replaced"
            )
        input_node.replace_all_uses_with(new_node)
        self.gm.graph.erase_node(input_node)
        clean_up_graph_after_modifications(self.gm)

    def rewrite_subgraph_nodes(self, subgraphs: List[ComplexSubGraphInfo]) -> None:
        modified = False
        for subgraph in subgraphs:
            for input_node in subgraph.input_nodes:
                logger.debug(f"Input node rewrite: {input_node.name}")
                if input_node.op not in ("call_function"):
                    self.replace_input_node(input_node)
            for node in subgraph.subgraph_nodes:
                logger.debug(f"Subgraph Node rewrite: {node.name}")
                if node.target == torch.ops.aten.view_as_complex.default:
                    node.replace_all_uses_with(node.args[0])
                    self.gm.graph.erase_node(node)
                elif node.target == torch.ops.aten.mul.Tensor:
                    # this is complex mul where inputs = a+ib and output = c+id.
                    # complex mul returns (ac - bd) + (ad + bc)i
                    # which is then view_as_real as (ac-bd), (ad+bc) stacked along the last dimension with last dimension size 2
                    x_placeholder_or_func = (
                        True if node.args[0].op != "get_attr" else False
                    )
                    y_placeholder_or_func = (
                        True if node.args[1].op != "get_attr" else False
                    )

                    replaced_nodes = []
                    original_mul, replacement = complex_mul_replacement(
                        x_placeholder_or_func, y_placeholder_or_func
                    )

                    def match_complex_mul(  # type: ignore[no-untyped-def]
                        match: torch.fx.subgraph_rewriter.Match,
                        original_graph,
                        pattern_graph,
                    ) -> bool:
                        for original_node in match.nodes_map.values():
                            if original_node.name == node.name:
                                return True
                        return False

                    nodes = torch.fx.subgraph_rewriter.replace_pattern_with_filters(
                        self.gm,
                        original_mul,
                        replacement,
                        match_filters=[match_complex_mul],
                        ignore_literals=True,
                    )
                    replaced_nodes += nodes
                    modified = True
                elif node.target == torch.ops.aten.view_as_real.default:
                    node.replace_all_uses_with(node.args[0])
                    self.gm.graph.erase_node(node)
                else:
                    logger.debug(f"Unsupported node target: {node.target}")
                    logger.debug(
                        "This complex subgraphnode type does not need to replaced"
                    )

        if modified:
            self.propagate_metadata()
            self.gm.graph.lint()
            self.gm.recompile()

    def propagate_metadata(self) -> None:
        fake_inputs = []
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.passes.fake_tensor_prop import FakeTensorProp

        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                if "val" in node.meta:
                    with FakeTensorMode(allow_non_fake_inputs=True):
                        fake_val = node.meta["val"]
                        fake_inputs.append(
                            fake_val.to("cuda")
                            if fake_val.device.type == "cuda"
                            else fake_val
                        )
                else:
                    fake_tensor = torch.empty(
                        [s if s != 0 else 1 for s in node.meta["tensor_meta"].shape],
                        dtype=node.meta["tensor_meta"].dtype,
                        device=node.meta["tensor_meta"].device,
                    )
                    fake_inputs.append(fake_tensor)
        FakeTensorProp(
            self.gm, mode=FakeTensorMode(allow_non_fake_inputs=True)
        ).propagate(*fake_inputs)


def extract_real_imag(input, placeholder_or_func: bool = True):  # type: ignore
    """Extract real and imaginary parts from a tensor.
    This function handles different tensor types based on whether they are placeholder/function
    tensors or get_attr tensors. For placeholder/function tensors, it uses select operations,
    while for get_attr tensors, it uses indexing.
    Args:
        input: Input tensor to extract real and imaginary parts from
        placeholder_or_func: Boolean flag indicating if the input is a placeholder/function tensor (True)
                           or a get_attr tensor (False). Defaults to True.
    Returns:
        Tuple of (real_part, imaginary_part) where both parts have the same type as the input
    Note:
        - When placeholder_or_func=True: Uses torch.ops.aten.select.int operations
        - When placeholder_or_func=False: Uses tensor indexing [..., 0] and [..., 1]
    """
    if placeholder_or_func:
        # For ITensor, use select operations
        real_part = torch.ops.aten.select.int(input, -1, 0)
        imag_part = torch.ops.aten.select.int(input, -1, 1)
        return real_part, imag_part
    else:
        # For get_attr, use indexing
        return input[..., 0], input[..., 1]


def complex_mul_replacement(
    x_placeholder_or_func: bool = True, y_placeholder_or_func: bool = True
) -> Tuple[
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
]:
    """Constructs the original and replacement functions for complex multiplication.

    The original functions correspond to native complex multiplication
    via torch.mul or operator.mul on complex tensors.

    The replacement function assumes x and y are real tensors with the last
    dimension size 2 representing real and imaginary parts, and performs
    complex multiplication manually returning the same shaped tensor.
    """

    # Original pattern: torch.mul for complex tensors
    def original_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.mul.Tensor(x, y)

    # Replacement function: manual complex multiplication on real/imag stacked tensors
    def replacement(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_real, x_imag = extract_real_imag(x, x_placeholder_or_func)
        y_real, y_imag = extract_real_imag(y, y_placeholder_or_func)

        real_part1 = torch.ops.aten.mul.Tensor(x_real, y_real)
        real_part2 = torch.ops.aten.mul.Tensor(x_imag, y_imag)
        real = torch.ops.aten.sub.Tensor(real_part1, real_part2)

        imag_part1 = torch.ops.aten.mul.Tensor(x_real, y_imag)
        imag_part2 = torch.ops.aten.mul.Tensor(x_imag, y_real)
        imag = torch.ops.aten.add.Tensor(imag_part1, imag_part2)

        return torch.ops.aten.cat.default(
            [
                torch.ops.aten.unsqueeze.default(real, -1),
                torch.ops.aten.unsqueeze.default(imag, -1),
            ],
            -1,
        )

    return (original_mul, replacement)


# This lowering pass is used to detect and rewrite complex subgraphs in the graph
def complex_graph_detection(
    gm: GraphModule, settings: CompilationSettings
) -> GraphModule:
    """Detect and rewrite complex subgraphs in the graph.
    This lowering pass is used to detect and rewrite complex subgraphs in the graph.
    This lowering pass works for complex tensor in mul which are parameter or buffers in the graph.
    Args:
        gm: The GraphModule to process
        settings: Compilation settings
    Returns:
        The modified GraphModule with complex subgraphs rewritten
    """
    complex_op_detector = ComplexOpDetector()
    complex_subgraphs = complex_op_detector.find_complex_op_subgraphs(
        gm, anchor_target=torch.ops.aten.view_as_real.default
    )
    for subgraph in complex_subgraphs:
        logger.debug(f"Complex subgraph info: {subgraph}")
    complex_graph_rewriter = ComplexGraphRewriter(gm, settings.truncate_double)
    complex_graph_rewriter.rewrite_subgraph_nodes(complex_subgraphs)
    return gm
