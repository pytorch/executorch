# EXIR Infrastructure

## Passes

Passes can be roughly categorized into a couple of axis:

Axis A:
1. Creating one-to-X mapping (eg. decomposition)
2. Creating many-to-one mapping (eg. fusion)

Axis B:
1. Doing forwards iteration (eg. shape propagation)
2. Doing backwards iteration (eg. dead code elimination)

Axis C:
1. Dependent on local node information (eg. out-variant conversion)
2. Dependent on global graph information (eg. memory planning)

Our projection on the frequency of these use cases are:
1. A.1, B.1, C.1
2. A.2
3. B.2, C.2

### Level 1

For level 1 uses cases (creating one-to-X mappings, doing forwards iterations,
and looking at local node information), we can utilize a helper class called
[`ExportPass`](https://fburl.com/code/ecf4kyax). This is an
[interpreter-based](https://pytorch.org/docs/stable/fx.html#the-interpreter-pattern)
way where we execute each node and recreate the graph except with
transformations specified. This allows us to preserve the IR Spec by ensuring
that all nodes created while in the pass meet the IR Spec such as making sure that
metadata such as stack trace, FakeTensor values, and torch.nn.Module heirarchy
are preserved and updated depending on the transformations made.

To implement this pass, we can subclass
[`ExportPass`](https://fburl.com/code/ecf4kyax) and implement the exposed
functions.  When called with a graph module, it will run the graph module and
create a new graph containing the changes specified by the pass. This means that
the graph module passed in must be runnable on CPU, and this invariant will be
maintained after the pass is run.

#### One-to-One Pass

An example for one-to-one mappings, if we wanted to replace an op A with another op B,
we can run the given
`ExportGraphModule`, and very time we see op A, return op B.

An example is:

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
can run the given `ExportGraphModule`, and for every argument that contains a scalar,
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
rewriter](https://fburl.com/code/9ld92myj). Given a `pattern`, it creates a
subgraph of operators matching to the pattern, and then replaces each matched
subgraph with the `replacement`.

Note:

    This is an inplace operation.

The `pattern` and `replacement` inputs must be callable functions written with
the same ops that are used in the EXIR graph you are matching with (ATen ops)
so that the subgraph rewriter can find the correct pattern in the graph. Inputs
to the pattern/replacement callables will be treated as wildcards.

An example:

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

Note:

    The nodes created by the subgraph rewriter will not have the metadata that
    is normally in EXIR nodes (`stack_trace`, `val`, `nn_module_stack`).


### Level 3

For the third way of creating a pass, we can utilize the most basic
[`PassBase`](https://fburl.com/code/qtuz4l47). To create a
pass, we can subclass this and implement the function `call` with the pass
contents. Additionally, we can implement the functions `requires` and `ensures`
which will be called before and after the function `call`. Note that these
functions can also be overridden in `ExportPass`. To run a pass on a
graph module, we can pass the graph module directly to an instance of the class.

An example:
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

There are a couple of common FX graph based partitioners we can use to partition
the graph. However, these do not necessarily produce a graph that is compliant
with IR Spec, so be careful when using them.

### Subgraph Matcher

For finding subgraphs within a graph that match a specific pattern, we can
utilize FX's [`SubgraphMatcher`](https://fburl.com/code/9ccshnvi).

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

An example:

```python
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher

class LargeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._weight = torch.nn.Parameter(torch.ones(3, 3))
        self._bias = torch.nn.Parameter(torch.ones(3, 3))

    def forward(self, x):
        return torch.ops.aten.addmm.default(self._bias, x, self._weight)

large_model_graph = exir.capture(LargeModel(), large_inputs).to_edge().graph

class PatternModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._weight_1 = torch.nn.Parameter(torch.ones(5, 5))
        self._bias_1 = torch.nn.Parameter(torch.ones(5, 5))

    def forward(self, x):
        return torch.ops.aten.addmm.default(self._bias_1, x, self._weight_1)

pattern_graph = exir.capture(PatternModel(), pattern_inputs).to_edge().graph

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
utilize FX's [`CapabilityBasedPartitioner`](https://fburl.com/code/hrw8h4r1).

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

The [`OperatorSupportBase`](https://fburl.com/code/xumdm1qc) class is used by
the partitioner to determine if a specific node in the graph belongs in the
partition. This is done by overriding the `is_node_supported` function. You can
chain multiple `OperatorSuppportBase` by using
[`chain`](https://fburl.com/code/cfmcj8bb)(which returns False if any of the
OperatorSupportBase return False) and
[`any_chain`](https://fburl.com/code/5bwe3364) (which returns True if any of the
OperatorSupportBase returns True).

An example:

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
[`generate_pattern_op_partitions`](https://fburl.com/code/pquxdl24)

Args:
* `graph_module (ExportGraphModule)`: Module that we want to partition
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
  given pattern graphs
