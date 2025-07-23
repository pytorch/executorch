# Custom Compiler Passes and Partitioners

## Passes

Passes can be roughly categorized into a couple of axes:

Axis A:
1. Creating one-to-X mapping (for example, decomposition)
2. Creating many-to-one mapping (for example, fusion)

Axis B:
1. Performing forwards iteration (for example, shape propagation)
2. Performing backwards iteration (for example, dead code elimination)

Axis C:
1. Dependent on local node information (eg. out-variant conversion)
2. Dependent on global graph information (eg. memory planning)

Our projection on the frequency of these use cases are:
1. A.1, B.1, C.1
2. A.2
3. B.2, C.2

### Level 1

For level 1 uses cases (creating one-to-X mappings, performing forwards iterations,
and looking at local node information), we can utilize a helper class called
[`ExportPass`](https://github.com/pytorch/executorch/blob/d9eef24bb720804aa7b400b05241487510ae0dc2/exir/pass_base.py#L44).
This is an
[interpreter-based](https://pytorch.org/docs/stable/fx.html#the-interpreter-pattern)
way where we execute each node and recreate the graph except with
transformations specified. This allows us to preserve the IR Spec by ensuring
that all nodes created while in the pass meet the IR Spec including ensuring that
metadata such as stack trace, FakeTensor values, and torch.nn.Module hierarchy
are preserved and updated depending on the transformations made.

To implement this pass, we can create a subclass of
[`ExportPass`](https://github.com/pytorch/executorch/blob/d9eef24bb720804aa7b400b05241487510ae0dc2/exir/pass_base.py#L44)
and implement the exposed functions.  When called with a graph module, it will
run the graph module and create a new graph containing the changes specified by
the pass. This means that the graph module passed in must be runnable on CPU,
and this invariant will be maintained after the pass is run.

#### One-to-One Pass

An example for one-to-one mappings, if we wanted to replace an op A with another op B,
we can run the given
`fx.GraphModule`, and every time we see op A, return op B.

Consider the following example:

```python
class ReplaceInPlaceReluWithOutOfPlaceReluPass(ExportPass):
    """
    relu_ is the in-place version. Replace it with relu, which is the
    out-of-place version
    """

    def call_operator(self, op, args, kwargs, meta):
        if op != torch.ops.aten.relu_.default:
            return super().call_operator(op, args, kwargs, meta)
        return super().call_operator(Op(torch.ops.aten.relu.default), args, kwargs, meta)

# To create a pass
replace_pass = ReplaceInPlaceReluWithOutOfPlaceReluPass()
# To run a pass
new_graph_module = replace_pass(graph_module).graph_module
```

The `super().call_operator(op, args, kwargs, meta)` call creates a
`call_function` FX node, and returns the result of running the operator with the
given arguments.

#### One-to-X Pass

If we wanted to do one-to-X mappings, like replacing op A with 2 other ops B and
C, we would then make 2 calls to `super().call_operator` to create 2 FX nodes,
one with op B and another with op C, and return the result of running op C.

For example:
```python
class ReplaceAddWithMulSub(ExportPass):
    """
    Original:
        def f(x, y):
            return x + y

    After pass:
        def f(x, y):
            z = x * y
            return z - y
    """
    def call_operator(self, op, args, kwargs, meta):
        if op != torch.ops.aten.add.default:
            return super().call_operator(op, args, kwargs, meta)

        x, y = args

        mul_res = super().call_operator(
            torch.ops.aten.mul.default,
            args,
            {},
            meta
        )

        return super().call_operator(
            torch.ops.aten.sub.default,
            (mul_res, y),
            {},
            meta
        )
```

#### One-to-None Pass

If we wanted to remove an op, we can just return the value passed into the
function:

```python
class RemoveDetachPass(ExportPass):
    def call_operator(self, op, args, kwargs, meta):
        if op not in (
            torch.ops.aten.detach.default,
            torch.ops.aten.detach_copy.default,
        ):
            return super().call_operator(op, args, kwargs, meta)

        assert len(args) == 1
        return args[0]
```

#### Utilizing Local Information

An example of utilizing local node information is, if we wanted to convert all the
scalars within the graph to tensors, we
can run the given `fx.GraphModule`, and for every argument that contains a scalar,
we convert it to a tensor. It might look something like:

```python
def args_map(op, fn, args, kwargs):
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    args = list(args)
    kwargs = kwargs.copy()

    # Update the argument based on the function passed
    def update(key, args, schema):
        args[key] = fn(args[key], schema)

    # Update each argument in the schema
    for i, schema in enumerate(self.op._schema.arguments):
        if schema.name in kwargs:
            update(schema.name, kwargs, schema)
        elif not schema.kwarg_only and i < len(args):
            update(i, args, schema)

class ScalarToTensorPass(ExportPass):
    def call_operator(self, op, args, kwargs):
        def try_coerce(value, arg):
            return (
                torch.tensor(value)
                if isinstance(value, (float, int, bool))
                and type(arg.type) == torch.TensorType
                else value
            )

        args, kwargs = args_map(op, try_coerce, args, kwargs)
        return super().call_operator(op, args, kwargs)
```

### Level 2

For creating many-to-one mappings, we can utilize FX's [subgraph
rewriter](https://github.com/pytorch/pytorch/blob/8597d37536ef11bdf6b0a539ab79af876e1c92f6/torch/fx/subgraph_rewriter.py#L77).
Given a `pattern`, it creates a subgraph of operators matching to the pattern,
and then replaces each matched subgraph with the `replacement`.

```{note}

    This is an inplace operation.

```

The `pattern` and `replacement` inputs must be callable functions written with
the same ops that are used in the EXIR graph you are matching with (ATen ops)
so that the subgraph rewriter can find the correct pattern in the graph. Inputs
to the pattern/replacement callables will be treated as wildcards.

Consider the following example:

```python
from torch.fx import subgraph_rewriter

def replace_patterns(graph_module):
    def pattern(x, y):
        x = torch.ops.aten.add.Tensor(x, y)
        x = torch.ops.aten.mul.Tensor(x, y)
        return x

    def replacement(x, y):
        return torch.ops.aten.sub.Tensor(x, y)

replaced_patterns = subgraph_rewriter.replace_pattern_with_filters(
    traced_module, pattern, replacement
)
```

The subgraph rewriter returns a list of `ReplacedPatterns`:

```python
@dataclass
class ReplacedPatterns:
    # Node from which the match was found
    anchor: Node
    # Maps nodes in the pattern subgraph to nodes in the larger graph
    nodes_map: Dict[Node, Node]
    # List of nodes that were added into the graph
    replacements: List[Node]
```

```{note}

    The nodes created by the subgraph rewriter will not have the metadata that
    is normally in EXIR nodes (`stack_trace`, `val`, `nn_module_stack`).

```


### Level 3

For the third way of creating a pass, we can utilize the most basic
[`PassBase`](https://github.com/pytorch/pytorch/blob/8597d37536ef11bdf6b0a539ab79af876e1c92f6/torch/fx/passes/infra/pass_base.py#L22).
To create a pass, we can subclass this and implement the function `call` with
the pass contents. Additionally, we can implement the functions `requires` and
`ensures` which will be called before and after the function `call`. Note that
these functions can also be overridden in `ExportPass`. To run a pass on a graph
module, we can pass the graph module directly to an instance of the class.

Consider the following example:

```python
class ReplaceAddPass(PassBase):

    def __init__(self, replace_op):
        self.replace_op = replace_op

    def call(self, graph_module):
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.add:
                node.target = self.replace_op

    # Optional to implement, will be called before call()
    def requires(self, graph_module) -> None:
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target == torch.add:
                return
        raise ValueError("No torch.add ops!")

    # Optional to implement, will be called after call()
    def ensures(self, graph_module: torch.fx.GraphModule) -> None:
        pass

# To create a pass
replace_add_with_div = ReplaceAddPass(torch.div)
# To run a pass
replace_add_with_div(graph_module)
```

## Pass Manager

The `PassManager` is a class used to run multiple passes on a given graph
module. When initializing a `PassManager` instance, we pass in a list of passes
that we want to run and set a couple of flags. To run the collection of passes
on a graph module, we can pass the graph module directly to the `PassManager`
instance.

An example:
```python
from executorch.exir.pass_manager import PassManager

pm = PassManager(
    passes=[replace_add_with_div, replace_div_with_mul],
    run_checks_after_each_pass=True,
    suppress_check_failures=False,
)
graph_module_out = pm(graph_module)
```

To add a common set of checks that are run after each pass, we can call the
function `set_checks(check: Callable)` which takes in a callable function as
input. If the `run_checks_after_each_pass` flag is set, the `check` will be
called after each pass is run on the graph module.

An example:
```python
pm = PassManager(passes=[replace_add_with_div, replace_div_with_mul])

def check_div_target(graph_module):
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and node.target != torch.div:
            raise ValueError("Target should be div!")

pm.add_checks(check_div_target)

pm(graph_module)    # raises ValueError after replace_div_with_mul pass
```

## Partitioner

There are a couple of common FX-graph based partitioners we can use to partition
the graph. However, these do not necessarily produce a graph that is compliant
with IR Spec, so be careful when using them.

### Subgraph Matcher

For finding subgraphs within a graph that match a specific pattern, we can
utilize FX's
[`SubgraphMatcher`](https://github.com/pytorch/pytorch/blob/8597d37536ef11bdf6b0a539ab79af876e1c92f6/torch/fx/passes/utils/matcher_utils.py#L51).

Class Attributes:

* `pattern (Graph)`: The targeted matching pattern. Placeholder nodes in the
   graph will be treated as wildcards when matching.
* `match_output (bool)`: If True, output node in the pattern graph will be
   treated as a part of the targeted pattern.  If False, output node is ignored
   during match.
* `match_placeholder (bool)`: If True, placeholder node in the pattern graph
   will be treated as a part of the targeted pattern. If False, placeholder
   nodes will be used a wildcard.
* `remove_overlapping_matches (bool)`: If True, in the case of overlapping
   matches, only the first match will be returned.
*  `ignore_literals (bool)`: If True, will not check if literals are equal and
   will instead treat them as wildcards.

Consider the following example:

```python
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher

class LargeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._weight = torch.nn.Parameter(torch.ones(3, 3))
        self._bias = torch.nn.Parameter(torch.ones(3, 3))

    def forward(self, x):
        return torch.ops.aten.addmm.default(self._bias, x, self._weight)

large_model_graph = to_edge(export(LargeModel(), large_inputs)).exported_program().graph_module.graph

class PatternModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._weight_1 = torch.nn.Parameter(torch.ones(5, 5))
        self._bias_1 = torch.nn.Parameter(torch.ones(5, 5))

    def forward(self, x):
        return torch.ops.aten.addmm.default(self._bias_1, x, self._weight_1)

pattern_graph = to_edge(export(PatternModel(), pattern_inputs)).exported_program().graph_module.graph

subgraph_matcher = SubgraphMatcher(pattern_graph)
match_result = subgraph_matcher.match(large_model_graph)
```

The `match` function returns a list of `InternalMatch`:

```python
@dataclass
class InternalMatch():
    # Nodes from which the match was found
    anchors: List[Node]
    # Maps nodes in the pattern subgraph to nodes in the larger graph
    nodes_map: Dict[Node, Node] = field(default_factory=dict)
    # Nodes in target graph that are matched placeholder in pattern
    placeholder_nodes: List[Node] = field(default_factory=list)
    # Nodes in matched subgraph returned by output
    returning_nodes: List[Node] = field(default_factory=list)
```

### Capability Based Partitioner

To find the largest subgraphs of nodes that support a specific invariant, we can
utilize FX's
[`CapabilityBasedPartitioner`](https://github.com/pytorch/pytorch/blob/8597d37536ef11bdf6b0a539ab79af876e1c92f6/torch/fx/passes/infra/partitioner.py#L34C1-L34C1).

Class Attributes

* `graph_module (torch.fx.GraphModule)`: The graph module we are partitioning on.
* `operator_support (OperatorSupportBase)`: The object used to determine if a
   node in the graph is supported in the partition.
* `allows_single_node_partition (bool)`: If True, allows single node
   partitions to be formed.
* `non_compute_ops (Optional[Sequence[str]])`: A set of ops that are
   considered to be "non-compute" (ex `torch.ops.aten.view` and
   `_operator.getitem`, so that the partitioner will not create graphs that only
   contain these non-compute ops
* `allowed_single_node_partition_ops (Optional[Sequence[str]])`: A set of ops
   that are allowed to be in a single node partition.

The
[`OperatorSupportBase`](https://github.com/pytorch/pytorch/blob/8597d37536ef11bdf6b0a539ab79af876e1c92f6/torch/fx/passes/operator_support.py#L28)
class is used by
the partitioner to determine if a specific node in the graph belongs in the
partition. This is done by overriding the `is_node_supported` function. You can
chain multiple `OperatorSuppportBase` by using
[`chain`](https://github.com/pytorch/pytorch/blob/8597d37536ef11bdf6b0a539ab79af876e1c92f6/torch/fx/passes/operator_support.py#L150)(which
returns False if any of the OperatorSupportBase return False) and
[`any_chain`](https://github.com/pytorch/pytorch/blob/8597d37536ef11bdf6b0a539ab79af876e1c92f6/torch/fx/passes/operator_support.py#L164)
(which returns True if any of the OperatorSupportBase returns True).

Consider the following example:

```python
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import any_chain, OperatorSupportBase

class AddMulOperatorSupport(OperatorSupportBase):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return node.op == "call_function" and node.target in [
            torch.ops.aten.add.Tensor, torch.ops.aten.mul.Tensor,
        ]

capability_partitioner = CapabilityBasedPartitioner(
    graph_module,
    op_support,
)

# Returns a list of partitions (list of nodes that belong in each partition)
partition_list = capability_partitioner.propose_partitions()
```

If you look at the capability based partitioner, you may also find a
`fuse_partition` function which will return a modified graph with the partitions
as submodules, and calls to these submodules in the toplevel graph through
`call_module` nodes. However, this is not compliant to the IR Spec because we do
not allow `call_module` nodes.


### Combined

We also provide a combined helper function:
[`generate_pattern_op_partitions`](https://github.com/pytorch/executorch/blob/d9eef24bb720804aa7b400b05241487510ae0dc2/exir/backend/canonical_partitioners/pattern_op_partitioner.py#L59)

Args:
* `graph_module (fx.GraphModule)`: Module that we want to partition
* `patterns (List[torch.fx.Graph])`: A list of patterns in the form of
   torch.fx.Graph. These graphs can be obtained through the `graph` field from a
   GraphModule obtained by exir.capture (recommended) or symbolic tracing (which
   might not result in an accurate edge dialect graph), or by manual crafting a
   graph module.
* `op_support (OperatorSupportBase)`: A OperatorSupportBase that can be created
   in the following ways:
    * Subclassing it directly and implementing `is_node_supported()`
    * Getting the result of `create_op_support()`
    * Getting the result of `create_pattern_support()`
    * Multiple OperatorSupportBase classes chained together with `chain()` or `any_chain()`

Returns
* A list of partitions (largest possible subgraphs) containing nodes are
  supported by the union of the given OperatorSupportBase object and the
  given pattern graphs.


### Source Partitioner

For more complicated use cases in which users want to partition based on higher
level modules (`torch.nn.Linear` or `torch.nn.functional.Linear`) which are now
decomposed into their operators (`aten.permute`, `aten.addmm`), we have the
following [helper function](https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/utils/source_matcher_utils.py#L51):

`get_source_partitions(graph: torch.fx.Graph, wanted_sources: List[Any]) -> Dict[Any, SourcePartition]`

Args:
* `graph`: The graph we want to partition
* `wanted_sources`: List of sources of nodes that were decomposed from this
    source. This can be a function (ex. `torch.nn.functional.linear`) or a leaf
    module type (ex. `torch.nn.Linear`)

Returns:
* Dictionary mapping sources (ex. `torch.nn.modules.linear.Linear`) to a list of
    `SourcePartitions` that correspond to the list of nodes that were flattened from
    a module of that type.

```python
@dataclass
class SourcePartition():
    # Nodes in a particular partition
    nodes: List[Node]
    # Module type
    module_type: Type
    # Nodes in the graph that are needed as inputs to the partition
    input_nodes: List[Node] = field(default_factory=list)
    # Nodes in the partition that are being used by nodes outside of the partition
    output_nodes: List[Node] = field(default_factory=list)
    # Parameters that are being used
    params: List[str] = field(default_factory=list)
```

An example:

```python
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 3)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(3, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

inputs = (torch.randn(3, 3),)
edge_graph = to_edge(export(M(), inputs)).exported_program().graph_module.graph
print(edge_graph)
"""
graph():
    %arg0 : [#users=1] = placeholder[target=arg0]
    %_param_constant0 : [#users=1] = get_attr[target=_param_constant0]
    %permute_default : [#users=1] = call_function[target=torch.ops.aten.permute_copy.default](args = (%_param_constant0,), kwargs = {})
    %_param_constant1 : [#users=1] = get_attr[target=_param_constant1]
    %addmm_default : [#users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%_param_constant1, %arg0, %t_default), kwargs = {})
    %_param_constant0_1 : [#users=1] = get_attr[target=_param_constant0]
    %permute_default_1 : [#users=1] = call_function[target=torch.ops.aten.permute_copy.default](args = (%_param_constant0_1,), kwargs = {})
    %_param_constant1_1 : [#users=1] = get_attr[target=_param_constant1]
    %addmm_default_1 : [#users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%_param_constant1_1, %addmm_default, %t_default_1), kwargs = {})
    %relu_default : [#users=1] = call_function[target=torch.ops.aten.relu.default](args = (%addmm_default_1,), kwargs = {})
    %_param_constant2 : [#users=1] = get_attr[target=_param_constant2]
    %permute_default_2 : [#users=1] = call_function[target=torch.ops.aten.permute_copy.default](args = (%_param_constant2,), kwargs = {})
    %_param_constant3 : [#users=1] = get_attr[target=_param_constant3]
    %addmm_default_2 : [#users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%_param_constant3, %relu_default, %t_default_2), kwargs = {})
    return [addmm_default_2]
"""

module_partitions = get_source_partitions(edge_graph, [torch.nn.Linear, torch.nn.ReLU])
print(module_partitions)
"""
{<class 'torch.nn.modules.linear.Linear'>: [
    ModulePartition(nodes=[_param_constant0, t_default, _param_constant1, addmm_default], module_type=<class 'torch.nn.modules.linear.Linear'>, input_nodes=[arg0], output_nodes=[addmm_default], params=["_param_constant0", "_param_constant1"]),
    ModulePartition(nodes=[_param_constant0_1, t_default_1, _param_constant1_1, addmm_default_1], module_type=<class 'torch.nn.modules.linear.Linear'>, input_nodes=[addmm_default], output_nodes=[addmm_default_1], params=["_param_constant0_1", "_param_constant1_1"]),
    ModulePartition(nodes=[_param_constant2, t_default_2, _param_constant3, addmm_default_2], module_type=<class 'torch.nn.modules.linear.Linear'>, input_nodes=[relu_default], output_nodes=[addmm_default_2], params=["_param_constant2", "_param_constant3"])],

 <class 'torch.nn.modules.activation.ReLU'>: [
    ModulePartition(nodes=[relu_default], module_type=<class 'torch.nn.modules.activation.ReLU'>, input_nodes=[addmm_default_1], output_nodes=[relu_default], params=[])]}
"""
```
