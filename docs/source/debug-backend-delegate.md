# Debug Backend Delegate

We provide a list of util functions to give users insights on what happened to the graph modules during the `to_backend()` stage.

## Get delegation summary
The `get_delegation_info()` method provides a summary of what happened to the model after the `to_backend()` call:

```python
from executorch.devtools.backend_debug import get_delegation_info
from tabulate import tabulate

# ... After call to to_backend(), but before to_executorch()
graph_module = edge_manager.exported_program().graph_module
delegation_info = get_delegation_info(graph_module)
print(delegation_info.get_summary())
df = delegation_info.get_operator_delegation_dataframe()
print(tabulate(df, headers="keys", tablefmt="fancy_grid"))
```

Example printout:
```
Total  delegated  subgraphs:  86
Number  of  delegated  nodes:  473
Number  of  non-delegated  nodes:  430
```


|    |  op_type                                 |  occurrences_in_delegated_graphs  |  occurrences_in_non_delegated_graphs  |
|----|---------------------------------|------- |-----|
|  0  |  aten__softmax_default  |  12  |  0  |
|  1  |  aten_add_tensor  |  37  |  0  |
|  2  |  aten_addmm_default  |  48  |  0  |
|  3  |  aten_arange_start_step  |  0  |  25  |
|      |  ...  |    |    |
|  23  |  aten_view_copy_default  |  170  |  48  |
|      |  ...  |    |    |
|  26  |  Total  |  473  |  430  |

From the table, the operator `aten_view_copy_default` appears 170 times in delegate graphs and 48 times in non-delegated graphs. Users can use information like this to debug.

## Visualize delegated graph
To see a more detailed view, use the `format_delegated_graph()` method to get a str of printout of the whole graph or use `print_delegated_graph()` to print directly:

```python
from executorch.exir.backend.utils import format_delegated_graph
graph_module = edge_manager.exported_program().graph_module
print(format_delegated_graph(graph_module)) # or call print_delegated_graph(graph_module)
```
It will print the whole model as well as the subgraph consumed by the backend. The generic debug function provided by fx like `print_tabular()` or `print_readable()` will only show `call_delegate` but hide the the subgraph consumes by the backend, while this function exposes the contents inside the subgraph.

In the example printout below, observe that `embedding` and `add` operators are delegated to `XNNPACK` while the `sub` operator is not.

```
%aten_unsqueeze_copy_default_22 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.unsqueeze_copy.default](args = (%aten_arange_start_step_23, -2), kwargs = {})
  %aten_unsqueeze_copy_default_23 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.unsqueeze_copy.default](args = (%aten_arange_start_step_24, -1), kwargs = {})
  %lowered_module_0 : [num_users=1] = get_attr[target=lowered_module_0]
    backend_id: XnnpackBackend
    lowered graph():
      %aten_embedding_default : [num_users=1] = placeholder[target=aten_embedding_default]
      %aten_embedding_default_1 : [num_users=1] = placeholder[target=aten_embedding_default_1]
      %aten_add_tensor : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%aten_embedding_default, %aten_embedding_default_1), kwargs = {})
      return (aten_add_tensor,)
  %executorch_call_delegate : [num_users=1] = call_function[target=torch.ops.higher_order.executorch_call_delegate](args = (%lowered_module_0, %aten_embedding_default, %aten_embedding_default_1), kwargs = {})
  %aten_sub_tensor : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.sub.Tensor](args = (%aten_unsqueeze_copy_default, %aten_unsqueeze_copy_default_1), kwargs = {})
```
