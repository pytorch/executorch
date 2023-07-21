from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from executorch.sdk.edir.base_schema import Node, OperatorGraph, OperatorNode, ValueNode
from executorch.sdk.edir.et_schema import PROFILE_STAT_HEADER, RESERVED_METADATA_ARG
from executorch.sdk.etdb.row_schema import (
    AbstractNodeInstanceRow,
    GraphInstanceRow,
    OpInstanceRow,
    OpSummaryRow,
    ValueInstanceRow,
)
from tabulate import tabulate


# Generate OpSummary Rows from Grouping by module type
def _gen_module_summaries(modules: List[GraphInstanceRow]) -> Dict[str, OpSummaryRow]:
    # Group by module type
    summaries = {}
    for module in modules:
        if (module_type := module.get_module_type()) not in summaries:
            summaries[module_type] = OpSummaryRow(module_type, [])
        summaries[module_type].elements.append(module)

    return summaries


# Generate OpSummary Rows from Grouping by operator type
# Extract stats from an aggregated ops summary table if provided
def _gen_op_summaries(
    ops: List[OpInstanceRow], aggr_op_stats: Optional[List[Tuple[Any, ...]]] = None
) -> Dict[str, OpSummaryRow]:
    # Group by operator type
    grouped_ops = {}
    for op in ops:
        if op.op not in grouped_ops:
            grouped_ops[op.op] = []
        grouped_ops[op.op].append(op)

    summaries = {}

    # Extract stats from the pregenerated table
    # Note: These fields can be opaquely extracted from the pregenerated table
    if aggr_op_stats is not None:
        header = aggr_op_stats[0]
        for row in aggr_op_stats[1:]:
            op_name = row[header.index(PROFILE_STAT_HEADER.NAME.value)]
            mean_ms = row[header.index(PROFILE_STAT_HEADER.MEAN_MS.value)]
            min_ms = row[header.index(PROFILE_STAT_HEADER.MIN_MS.value)]
            p10_ms = row[header.index(PROFILE_STAT_HEADER.P10_MS.value)]
            p90_ms = row[header.index(PROFILE_STAT_HEADER.P90_MS.value)]
            max_ms = row[header.index(PROFILE_STAT_HEADER.MAX_MS.value)]
            summaries[op_name] = OpSummaryRow(
                op_name, grouped_ops[op_name], mean_ms, min_ms, p10_ms, p90_ms, max_ms
            )
    else:
        for name, grouped_op in grouped_ops.items():
            summaries[name] = OpSummaryRow(name, grouped_op)

    return summaries


# Look for specific Pregenerated Tables, returning None if not found
def _extract_pre_generated_tables(metadata: Dict[str, Any]):
    return (
        metadata.get("tables", {}).get(RESERVED_METADATA_ARG.AGGREGATED_OP_TABLE.value),
        metadata.get("tables", {}).get(RESERVED_METADATA_ARG.RUN_SUMMARY_TABLE.value),
    )


# Given a list of InstanceRows, populate their output_nodes fields based on the input nodes
# of the other InstanceRows
def _populate_outputs(
    instances_with_inputs: List[Union[OpInstanceRow, ValueInstanceRow]],
    operator_instances: Dict[str, OpInstanceRow],
    constant_instances: Dict[str, ValueInstanceRow],
    input_instances: Dict[str, ValueInstanceRow],
):
    for row in instances_with_inputs:
        for input_node in row.input_nodes:
            if input_node in operator_instances:
                operator_instances[input_node].output_nodes.append(row.name)
            elif input_node in constant_instances:
                constant_instances[input_node].output_nodes.append(row.name)
            elif input_node in input_instances:
                input_instances[input_node].output_nodes.append(row.name)


# Print out all Initial ET graph tables
def _print_all(
    input_instances: Dict[str, ValueInstanceRow],
    operator_instances: Dict[str, Union[OpInstanceRow, GraphInstanceRow]],
    operator_summary: Dict[str, OpSummaryRow],
    output_instances: Dict[str, ValueInstanceRow],
    verbose: bool = False,
):
    print("Inputs")
    print_rows(list(input_instances.values()), verbose)

    print("Operators")
    print_rows(list(operator_instances.values()), verbose)

    print("Aggregated Operator/Module Summaries")
    print_rows(list(operator_summary.values()), verbose)

    print("Outputs")
    print_rows(list(output_instances.values()), verbose)


# Pyre is being weird with subclass typing:
#   rows is List[AbstractInstanceRow] such that each row is the same subtype
#
# Give a list of AbstractInstanceRows, print in a table format
def print_rows(rows: List[Any], verbose: bool = False):
    if len(rows) > 0:
        print_table(
            type(rows[0]).get_schema_header(verbose),
            [entry.to_row_format(verbose) for entry in rows],
        )


# Table format used in ETDB
def print_table(header: List[str], rows: List[Sequence[Any]]):
    empty_columns = [False] * len(header)
    # Drop Columns with all None
    for index in range(len(header)):
        if all(
            (index >= len(row) or row[index] is None or row[index] == "")
            for row in rows
        ):
            empty_columns[index] = True

    header = [val for index, val in enumerate(header) if not empty_columns[index]]
    rows = [
        [val for index, val in enumerate(row) if not empty_columns[index]]
        for row in rows
    ]

    print(tabulate(rows, headers=header, tablefmt="fancy_grid"))


# Given a list of row identifing strings, print out the rows
# within corresponding tables
def _print_related_rows(
    inputs: List[str],
    input_instances: Dict[str, ValueInstanceRow],
    operator_instances: Dict[str, OpInstanceRow],
    constant_instances: Dict[str, ValueInstanceRow],
    operator_summary: Dict[str, OpSummaryRow],
    output_instances: Dict[str, ValueInstanceRow],
    subgraph_instances: Dict[str, GraphInstanceRow],
    verbose: bool = False,
):
    ops = []
    consts = []
    model_vals = []
    modules = []
    for in_arg in inputs:
        if in_arg in operator_instances:
            ops.append(operator_instances[in_arg])
        elif in_arg in constant_instances:
            consts.append(constant_instances[in_arg])
        elif in_arg in input_instances:
            model_vals.append(input_instances[in_arg])
        elif in_arg in output_instances:
            model_vals.append(output_instances[in_arg])
        elif in_arg in subgraph_instances:
            modules.append(subgraph_instances[in_arg])

    tables = {
        "Modules": modules,
        "Operators": ops,
        "Constants": consts,
        "Model Values": model_vals,
    }
    for name, rows in tables.items():
        if len(rows) > 0:
            print(name)
            print_rows(rows, verbose)


# Evaluate the request if it is asking for a backstep in history
def _eval_backstep(
    target: str,
    history: List[str],
    input_instances: Dict[str, ValueInstanceRow],
    operator_instances: Dict[str, OpInstanceRow],
    constant_instances: Dict[str, ValueInstanceRow],
    operator_summary: Dict[str, OpSummaryRow],
    output_instances: Dict[str, ValueInstanceRow],
    subgraph_instances: Dict[str, GraphInstanceRow],
    verbose: bool = False,
) -> bool:
    back_representations = {"Back", "back", "b"}
    if target not in back_representations:
        return False

    if len(history) <= 1:
        print("No history found")
    else:
        history.pop()
        _eval(
            history.pop(),
            history,
            input_instances,
            operator_instances,
            constant_instances,
            operator_summary,
            output_instances,
            subgraph_instances,
            verbose,
        )

    return True


# Evaluate the request if it is asking for an operator or module summary
def _eval_op_summary(
    target: str,
    history: List[str],
    operator_summary: Dict[str, OpSummaryRow],
    verbose: bool = False,
) -> bool:
    if (
        target not in operator_summary
        and (target := "[Sub Module] " + target) not in operator_summary
    ):
        return False

    selection = operator_summary[target]

    print("\nSelection\n---------")
    print_rows([selection], verbose)

    print("\nElements\n---------")
    if len(selection.elements) > 0:
        print("Forward")
        print_rows(selection.elements, verbose)

    history.append(target)

    return True


# Parse and perform a single pass of debug printing given a target input
# Request Types:
#  - Backstep in History
#  - Model Value (Input, Output)
#  - Op Instance
#  - Op Summary
#  - Constant Instance
def _eval(  # noqa C901
    target: str,
    history: List[str],
    input_instances: Dict[str, ValueInstanceRow],
    operator_instances: Dict[str, OpInstanceRow],
    constant_instances: Dict[str, ValueInstanceRow],
    operator_summary: Dict[str, OpSummaryRow],
    output_instances: Dict[str, ValueInstanceRow],
    subgraph_instances: Dict[str, GraphInstanceRow],
    verbose: bool = False,
):
    # Evaluate if request is to backstep in history
    if _eval_backstep(
        target,
        history,
        input_instances,
        operator_instances,
        constant_instances,
        operator_summary,
        output_instances,
        subgraph_instances,
        verbose,
    ):
        return

    # Evaluate if request is for an op summary
    if _eval_op_summary(target, history, operator_summary, verbose):
        return

    # Check for valid inputs
    selection = None
    sources = [
        input_instances,
        constant_instances,
        output_instances,
        operator_instances,
        subgraph_instances,
    ]
    for source in sources:
        if target in source:
            selection = source[target]

    # Valid input not found
    if selection is None:
        print("Invalid Input")
        return

    print("\nSelection\n---------")
    print_rows([selection], verbose)

    def _curried_print_tables(inputs: List[str]):
        _print_related_rows(
            inputs,
            input_instances,
            operator_instances,
            constant_instances,
            operator_summary,
            output_instances,
            subgraph_instances,
            verbose,
        )

    if target in subgraph_instances:
        print("\nElements\n------")
        # pyre-ignore isinstance doesn't play well with imported dataclasses
        _curried_print_tables(selection.elements)

    input_nodes = selection.input_nodes
    if len(input_nodes) > 0:
        print("\nInputs\n------")
        _curried_print_tables(input_nodes)

    output_nodes = selection.output_nodes
    if len(output_nodes) > 0:
        print("\nOutputs\n-------")
        _curried_print_tables(output_nodes)

    module = selection.parent_graph
    if module is not None:
        print("\nParent Module\n-------------")
        print_rows([subgraph_instances[module]], verbose)

    # Op Summary
    if isinstance(selection, OpInstanceRow):
        print("\nOperator Summary\n----------------")
        print_rows([operator_summary[selection.op]], verbose)

    if isinstance(selection, GraphInstanceRow):
        print("\nModule Summary\n--------------")
        print_rows([operator_summary[selection.get_module_type()]], verbose)

    history.append(target)


# Loop containing the interactive debugging state and processing
def enter_interactive_debugging(
    input_instances: Dict[str, ValueInstanceRow],
    operator_instances: Dict[str, OpInstanceRow],
    constant_instances: Dict[str, ValueInstanceRow],
    operator_summary: Dict[str, OpSummaryRow],
    output_instances: Dict[str, ValueInstanceRow],
    subgraph_instances: Dict[str, GraphInstanceRow],
    verbose: bool = False,
):
    history = []

    while True:
        print(
            "\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        )
        target = input(
            "Select one of the following (Node/Module Type, Node/Module Instance, Constant Name):\n> "
        )
        _eval(
            target,
            history,
            input_instances,
            operator_instances,
            constant_instances,
            operator_summary,
            output_instances,
            subgraph_instances,
            verbose,
        )


# Select one of the graphs for debugging
# TODO: Add the ability to toggle between graphs
def debug_graphs(graphs: Mapping[str, OperatorGraph], verbose: bool = False):
    print("Graphs: ", "\t".join(graphs.keys()))
    target = input("Select a graph to investigate:\n> ")
    if target not in graphs:
        target = input("Invalid Selection, Please try again:\n> ")

    debug_graph(graphs[target], verbose)


# Entry point for interactive debugging via ETDB
# Complexity lint, will be fixed in refactor
def debug_graph(graph: OperatorGraph, verbose: bool = False):  # noqa C901
    # Visual Separator
    print("\n")

    metadata = graph.metadata
    aggregated_op_table, run_summary_table = None, None
    if metadata is not None:
        # Extract Pregenerated Tables
        aggregated_op_table, run_summary_table = _extract_pre_generated_tables(metadata)

    # High level tables
    top_graph_instances = [
        GraphInstanceRow.gen_from_operator_node(element)
        for element in graph.elements
        if isinstance(element, OperatorGraph)
    ]

    if len(top_graph_instances) != len(graph.elements):
        raise RuntimeError(
            "Mixing Nodes and Graphs within OperatorGraph currently unsupported"
        )

    if run_summary_table is not None:
        print("Run Summary Table")
        print_table(run_summary_table[0], run_summary_table[1:])

    # Print High Level Tables
    print_table(
        GraphInstanceRow.get_schema_header(verbose, count_format=True),
        [
            entry.to_row_format(verbose, count_format=True)
            for entry in top_graph_instances
        ],
    )

    # Construct rows
    input_instances: Dict[str, ValueInstanceRow] = {}
    output_instances: Dict[str, ValueInstanceRow] = {}
    operator_instances: Dict[str, OpInstanceRow] = {}
    constant_instances: Dict[str, ValueInstanceRow] = {}
    subgraph_instances: Dict[str, GraphInstanceRow] = {}

    # Given a string identifier, return the corresponding NodeInstanceRow
    def find_instance_row(name: str) -> AbstractNodeInstanceRow:
        sources = [
            input_instances,
            output_instances,
            operator_instances,
            constant_instances,
            subgraph_instances,
        ]
        for source in sources:
            if name in source:
                return source[name]

        raise Exception(f"Could not find row identified with {name}")

    # Convert the provided Node/OperatorGraph into an InstanceRow and update the
    # corresponding data structures
    def add_row_instance(
        element: Union[Node, OperatorGraph], parent: Optional[str] = None
    ) -> None:
        if isinstance(element, ValueNode):
            row = ValueInstanceRow.gen_from_operator_node(element, parent)
            if sub_graph.graph_name == "inputs":
                input_instances[element.name] = row
            elif sub_graph.graph_name == "outputs":
                output_instances[element.name] = row
            else:
                constant_instances[element.name] = row
        elif isinstance(element, OperatorNode):
            row = operator_instances[
                element.name
            ] = OpInstanceRow.gen_from_operator_node(element, parent)
        elif isinstance(element, OperatorGraph):
            row = subgraph_instances[
                element.graph_name
            ] = GraphInstanceRow.gen_from_operator_node(element, parent)
            for child in element.elements:
                add_row_instance(child, parent=element.graph_name)
        else:
            raise RuntimeError(
                "Mixing Nodes and Graphs within OperatorGraph currently unsupported"
            )

    for sub_graph in graph.elements:
        # Enforced above
        assert isinstance(sub_graph, OperatorGraph)
        for element in sub_graph.elements:
            add_row_instance(element)

    # Populate Output
    instances_with_inputs = list(output_instances.values()) + list(
        operator_instances.values()
    )
    _populate_outputs(
        instances_with_inputs, operator_instances, constant_instances, input_instances
    )

    def collect_recursively(
        instance_node: AbstractNodeInstanceRow,
        graph_fn: Callable[[GraphInstanceRow], List[str]],
        general_fn: Callable[[AbstractNodeInstanceRow], List[str]],
    ) -> Set[str]:
        """
        Recursively collect the results of applying graph_fn/general_fn on
        a NodeInstanceRow

        - If instance_node is not a subgraph: return the result of general_fn(node)
        - If instance_node is a subgraph: return the result of graph_fn(node) plus the
        recursive output of this functions on node.elements

        Returns a set of unique strings curated from recursing through the provided node
        """
        collection = set()

        if instance_node.get_name() in subgraph_instances:
            # pyre-ignore isinstance doesn't play well with imported dataclasses
            collection.update(graph_fn(instance_node))
            # pyre-ignore
            for node in instance_node.elements:
                collection.update(
                    collect_recursively(find_instance_row(node), graph_fn, general_fn)
                )
        else:
            collection.update(general_fn(instance_node))

        return collection

    # Populate Input/Output of subgraphs
    for sub_graph in subgraph_instances.values():
        descendents = collect_recursively(
            sub_graph, (lambda node: node.elements), (lambda node: [])
        )
        descendent_inputs = collect_recursively(
            sub_graph,
            (lambda node: node.get_input_nodes()),
            (lambda node: node.get_input_nodes()),
        )
        descendent_outputs = collect_recursively(
            sub_graph,
            (lambda node: node.get_output_nodes()),
            (lambda node: node.get_output_nodes()),
        )

        true_inputs = descendent_inputs - descendents
        true_outputs = descendent_outputs - descendents

        sub_graph.input_nodes = list(true_inputs)
        sub_graph.output_nodes = list(true_outputs)

    # Generate Summary Table
    operator_summary = {
        **_gen_op_summaries(list(operator_instances.values()), aggregated_op_table),
        **{**_gen_module_summaries(list(subgraph_instances.values()))},
    }

    # Replace Operators with their parent groups
    collapsed_operators = {
        **{
            op_str: op
            for op_str, op in operator_instances.items()
            if op.parent_graph is None
        },
        **{
            op_str: op
            for op_str, op in subgraph_instances.items()
            if op.parent_graph is None
        },
    }

    # Print out tables
    _print_all(
        input_instances,
        collapsed_operators,
        operator_summary,
        output_instances,
        verbose,
    )

    # Start interactive debugging mode
    enter_interactive_debugging(
        input_instances,
        operator_instances,
        constant_instances,
        operator_summary,
        output_instances,
        subgraph_instances,
        verbose,
    )
