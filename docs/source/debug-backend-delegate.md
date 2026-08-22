# Debugging Delegation

We provide a list of util functions to give users insights on what happened to the graph modules during the `to_backend()` stage.

## Get delegation summary
The `get_delegation_info()` method provides a summary of what happened to the model after the `to_backend()` call:

```python
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import Dim, export
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
import torchvision.models as models

# Dependency needed for debugging delegates
from executorch.devtools.backend_debug import get_delegation_info
from tabulate import tabulate


model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

et_program = to_edge_transform_and_lower(
    torch.export.export(model, sample_inputs),
    partitioner=[XnnpackPartitioner()]
)
graph_module = et_program.exported_program().graph_module
delegation_info = get_delegation_info(graph_module)
# print the summary like the number of delegated nodes, non-delegated nodes, etc
print(delegation_info.get_summary())
df = delegation_info.get_operator_delegation_dataframe()
# print the table including op_type, occurrences_in_delegated_graphs, occurrences_in_non_delegated_graphs
print(tabulate(df, headers="keys", tablefmt="fancy_grid"))
```

Example printout:
```
Total delegated subgraphs: 2
Number of delegated nodes: 203
Number of non-delegated nodes: 4
```

|    | op_type                                           | occurrences_in_delegated_graphs | occurrences_in_non_delegated_graphs |
|----|---------------------------------------------------|---------------------------------|-------------------------------------|
|  0 | aten__native_batch_norm_legit_no_training_default | 52                              | 0                                   |
|  1 | aten_add_tensor                                   | 10                              | 0                                   |
|  2 | aten_convolution_default                          | 52                              | 0                                   |
|  3 | aten_hardtanh_default                             | 35                              | 0                                   |
|  4 | aten_linear_default                               | 1                               | 0                                   |
|  5 | aten_mean_dim                                     | 1                               | 0                                   |
|  6 | aten_view_copy_default                            | 0                               | 1                                   |
|  7 | dim_order_ops__clone_dim_order_default            | 0                               | 1                                   |
|  8 | getitem                                           | 52                              | 2                                   |
|  9 | **Total**                                         | **203**                         | **4**                               |


From the table, the operator `aten_view_copy_default` appears 0 times in delegate graphs and 1 times in non-delegated graphs. Users can use information like this to debug. `get_item node` is a special case, it means getting the output from the delegate subgraph.

## Visualize delegated graph
To see a more detailed view, use the `format_delegated_graph()` method to get a string representation of the entire graph or use `print_delegated_graph()` to print directly:

```python
from executorch.exir.backend.utils import format_delegated_graph
graph_module = et_program.exported_program().graph_module
print(format_delegated_graph(graph_module)) # or call print_delegated_graph(graph_module)
```
It will print the whole model as well as the subgraph consumed by the backend. The generic debug function provided by fx like `print_tabular()` or `print_readable()` will only show `call_delegate` and hide the subgraph consumed by the backend, while this function exposes the contents inside the subgraph.

In the example printout below, observe that there are two subgraphs, `aten_view_copy_default` is not delegated, while most of the others ops are delegated.

<details>
```
graph():
  %b_features_0_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_0_1_num_batches_tracked]
  %b_features_1_conv_0_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_1_conv_0_1_num_batches_tracked]
  %b_features_1_conv_2_num_batches_tracked : [num_users=0] = placeholder[target=b_features_1_conv_2_num_batches_tracked]
  %b_features_2_conv_0_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_2_conv_0_1_num_batches_tracked]
  %b_features_2_conv_1_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_2_conv_1_1_num_batches_tracked]
  %b_features_2_conv_3_num_batches_tracked : [num_users=0] = placeholder[target=b_features_2_conv_3_num_batches_tracked]
  %b_features_3_conv_0_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_3_conv_0_1_num_batches_tracked]
  %b_features_3_conv_1_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_3_conv_1_1_num_batches_tracked]
  %b_features_3_conv_3_num_batches_tracked : [num_users=0] = placeholder[target=b_features_3_conv_3_num_batches_tracked]
  %b_features_4_conv_0_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_4_conv_0_1_num_batches_tracked]
  %b_features_4_conv_1_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_4_conv_1_1_num_batches_tracked]
  %b_features_4_conv_3_num_batches_tracked : [num_users=0] = placeholder[target=b_features_4_conv_3_num_batches_tracked]
  %b_features_5_conv_0_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_5_conv_0_1_num_batches_tracked]
  %b_features_5_conv_1_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_5_conv_1_1_num_batches_tracked]
  %b_features_5_conv_3_num_batches_tracked : [num_users=0] = placeholder[target=b_features_5_conv_3_num_batches_tracked]
  %b_features_6_conv_0_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_6_conv_0_1_num_batches_tracked]
  %b_features_6_conv_1_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_6_conv_1_1_num_batches_tracked]
  %b_features_6_conv_3_num_batches_tracked : [num_users=0] = placeholder[target=b_features_6_conv_3_num_batches_tracked]
  %b_features_7_conv_0_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_7_conv_0_1_num_batches_tracked]
  %b_features_7_conv_1_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_7_conv_1_1_num_batches_tracked]
  %b_features_7_conv_3_num_batches_tracked : [num_users=0] = placeholder[target=b_features_7_conv_3_num_batches_tracked]
  %b_features_8_conv_0_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_8_conv_0_1_num_batches_tracked]
  %b_features_8_conv_1_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_8_conv_1_1_num_batches_tracked]
  %b_features_8_conv_3_num_batches_tracked : [num_users=0] = placeholder[target=b_features_8_conv_3_num_batches_tracked]
  %b_features_9_conv_0_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_9_conv_0_1_num_batches_tracked]
  %b_features_9_conv_1_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_9_conv_1_1_num_batches_tracked]
  %b_features_9_conv_3_num_batches_tracked : [num_users=0] = placeholder[target=b_features_9_conv_3_num_batches_tracked]
  %b_features_10_conv_0_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_10_conv_0_1_num_batches_tracked]
  %b_features_10_conv_1_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_10_conv_1_1_num_batches_tracked]
  %b_features_10_conv_3_num_batches_tracked : [num_users=0] = placeholder[target=b_features_10_conv_3_num_batches_tracked]
  %b_features_11_conv_0_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_11_conv_0_1_num_batches_tracked]
  %b_features_11_conv_1_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_11_conv_1_1_num_batches_tracked]
  %b_features_11_conv_3_num_batches_tracked : [num_users=0] = placeholder[target=b_features_11_conv_3_num_batches_tracked]
  %b_features_12_conv_0_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_12_conv_0_1_num_batches_tracked]
  %b_features_12_conv_1_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_12_conv_1_1_num_batches_tracked]
  %b_features_12_conv_3_num_batches_tracked : [num_users=0] = placeholder[target=b_features_12_conv_3_num_batches_tracked]
  %b_features_13_conv_0_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_13_conv_0_1_num_batches_tracked]
  %b_features_13_conv_1_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_13_conv_1_1_num_batches_tracked]
  %b_features_13_conv_3_num_batches_tracked : [num_users=0] = placeholder[target=b_features_13_conv_3_num_batches_tracked]
  %b_features_14_conv_0_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_14_conv_0_1_num_batches_tracked]
  %b_features_14_conv_1_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_14_conv_1_1_num_batches_tracked]
  %b_features_14_conv_3_num_batches_tracked : [num_users=0] = placeholder[target=b_features_14_conv_3_num_batches_tracked]
  %b_features_15_conv_0_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_15_conv_0_1_num_batches_tracked]
  %b_features_15_conv_1_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_15_conv_1_1_num_batches_tracked]
  %b_features_15_conv_3_num_batches_tracked : [num_users=0] = placeholder[target=b_features_15_conv_3_num_batches_tracked]
  %b_features_16_conv_0_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_16_conv_0_1_num_batches_tracked]
  %b_features_16_conv_1_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_16_conv_1_1_num_batches_tracked]
  %b_features_16_conv_3_num_batches_tracked : [num_users=0] = placeholder[target=b_features_16_conv_3_num_batches_tracked]
  %b_features_17_conv_0_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_17_conv_0_1_num_batches_tracked]
  %b_features_17_conv_1_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_17_conv_1_1_num_batches_tracked]
  %b_features_17_conv_3_num_batches_tracked : [num_users=0] = placeholder[target=b_features_17_conv_3_num_batches_tracked]
  %b_features_18_1_num_batches_tracked : [num_users=0] = placeholder[target=b_features_18_1_num_batches_tracked]
  %x : [num_users=1] = placeholder[target=x]
  %lowered_module_0 : [num_users=1] = get_attr[target=lowered_module_0]
    backend_id: XnnpackBackend
    lowered graph():
      %p_features_0_0_weight : [num_users=1] = placeholder[target=p_features_0_0_weight]
      %p_features_0_1_weight : [num_users=1] = placeholder[target=p_features_0_1_weight]
      %p_features_0_1_bias : [num_users=1] = placeholder[target=p_features_0_1_bias]
      %p_features_1_conv_0_0_weight : [num_users=1] = placeholder[target=p_features_1_conv_0_0_weight]
      %p_features_1_conv_0_1_weight : [num_users=1] = placeholder[target=p_features_1_conv_0_1_weight]
      %p_features_1_conv_0_1_bias : [num_users=1] = placeholder[target=p_features_1_conv_0_1_bias]
      %p_features_1_conv_1_weight : [num_users=1] = placeholder[target=p_features_1_conv_1_weight]
      %p_features_1_conv_2_weight : [num_users=1] = placeholder[target=p_features_1_conv_2_weight]
      %p_features_1_conv_2_bias : [num_users=1] = placeholder[target=p_features_1_conv_2_bias]
      %p_features_2_conv_0_0_weight : [num_users=1] = placeholder[target=p_features_2_conv_0_0_weight]
      %p_features_2_conv_0_1_weight : [num_users=1] = placeholder[target=p_features_2_conv_0_1_weight]
      %p_features_2_conv_0_1_bias : [num_users=1] = placeholder[target=p_features_2_conv_0_1_bias]
      %p_features_2_conv_1_0_weight : [num_users=1] = placeholder[target=p_features_2_conv_1_0_weight]
      %p_features_2_conv_1_1_weight : [num_users=1] = placeholder[target=p_features_2_conv_1_1_weight]
      %p_features_2_conv_1_1_bias : [num_users=1] = placeholder[target=p_features_2_conv_1_1_bias]
      %p_features_2_conv_2_weight : [num_users=1] = placeholder[target=p_features_2_conv_2_weight]
      %p_features_2_conv_3_weight : [num_users=1] = placeholder[target=p_features_2_conv_3_weight]
      %p_features_2_conv_3_bias : [num_users=1] = placeholder[target=p_features_2_conv_3_bias]
      %p_features_3_conv_0_0_weight : [num_users=1] = placeholder[target=p_features_3_conv_0_0_weight]
      %p_features_3_conv_0_1_weight : [num_users=1] = placeholder[target=p_features_3_conv_0_1_weight]
      %p_features_3_conv_0_1_bias : [num_users=1] = placeholder[target=p_features_3_conv_0_1_bias]
      %p_features_3_conv_1_0_weight : [num_users=1] = placeholder[target=p_features_3_conv_1_0_weight]
      %p_features_3_conv_1_1_weight : [num_users=1] = placeholder[target=p_features_3_conv_1_1_weight]
      %p_features_3_conv_1_1_bias : [num_users=1] = placeholder[target=p_features_3_conv_1_1_bias]
      %p_features_3_conv_2_weight : [num_users=1] = placeholder[target=p_features_3_conv_2_weight]
      %p_features_3_conv_3_weight : [num_users=1] = placeholder[target=p_features_3_conv_3_weight]
      %p_features_3_conv_3_bias : [num_users=1] = placeholder[target=p_features_3_conv_3_bias]
      %p_features_4_conv_0_0_weight : [num_users=1] = placeholder[target=p_features_4_conv_0_0_weight]
      %p_features_4_conv_0_1_weight : [num_users=1] = placeholder[target=p_features_4_conv_0_1_weight]
      %p_features_4_conv_0_1_bias : [num_users=1] = placeholder[target=p_features_4_conv_0_1_bias]
      %p_features_4_conv_1_0_weight : [num_users=1] = placeholder[target=p_features_4_conv_1_0_weight]
      %p_features_4_conv_1_1_weight : [num_users=1] = placeholder[target=p_features_4_conv_1_1_weight]
      %p_features_4_conv_1_1_bias : [num_users=1] = placeholder[target=p_features_4_conv_1_1_bias]
      %p_features_4_conv_2_weight : [num_users=1] = placeholder[target=p_features_4_conv_2_weight]
      %p_features_4_conv_3_weight : [num_users=1] = placeholder[target=p_features_4_conv_3_weight]
      %p_features_4_conv_3_bias : [num_users=1] = placeholder[target=p_features_4_conv_3_bias]
      %p_features_5_conv_0_0_weight : [num_users=1] = placeholder[target=p_features_5_conv_0_0_weight]
      %p_features_5_conv_0_1_weight : [num_users=1] = placeholder[target=p_features_5_conv_0_1_weight]
      %p_features_5_conv_0_1_bias : [num_users=1] = placeholder[target=p_features_5_conv_0_1_bias]
      %p_features_5_conv_1_0_weight : [num_users=1] = placeholder[target=p_features_5_conv_1_0_weight]
      %p_features_5_conv_1_1_weight : [num_users=1] = placeholder[target=p_features_5_conv_1_1_weight]
      %p_features_5_conv_1_1_bias : [num_users=1] = placeholder[target=p_features_5_conv_1_1_bias]
      %p_features_5_conv_2_weight : [num_users=1] = placeholder[target=p_features_5_conv_2_weight]
      %p_features_5_conv_3_weight : [num_users=1] = placeholder[target=p_features_5_conv_3_weight]
      %p_features_5_conv_3_bias : [num_users=1] = placeholder[target=p_features_5_conv_3_bias]
      %p_features_6_conv_0_0_weight : [num_users=1] = placeholder[target=p_features_6_conv_0_0_weight]
      %p_features_6_conv_0_1_weight : [num_users=1] = placeholder[target=p_features_6_conv_0_1_weight]
      %p_features_6_conv_0_1_bias : [num_users=1] = placeholder[target=p_features_6_conv_0_1_bias]
      %p_features_6_conv_1_0_weight : [num_users=1] = placeholder[target=p_features_6_conv_1_0_weight]
      %p_features_6_conv_1_1_weight : [num_users=1] = placeholder[target=p_features_6_conv_1_1_weight]
      %p_features_6_conv_1_1_bias : [num_users=1] = placeholder[target=p_features_6_conv_1_1_bias]
      %p_features_6_conv_2_weight : [num_users=1] = placeholder[target=p_features_6_conv_2_weight]
      %p_features_6_conv_3_weight : [num_users=1] = placeholder[target=p_features_6_conv_3_weight]
      %p_features_6_conv_3_bias : [num_users=1] = placeholder[target=p_features_6_conv_3_bias]
      %p_features_7_conv_0_0_weight : [num_users=1] = placeholder[target=p_features_7_conv_0_0_weight]
      %p_features_7_conv_0_1_weight : [num_users=1] = placeholder[target=p_features_7_conv_0_1_weight]
      %p_features_7_conv_0_1_bias : [num_users=1] = placeholder[target=p_features_7_conv_0_1_bias]
      %p_features_7_conv_1_0_weight : [num_users=1] = placeholder[target=p_features_7_conv_1_0_weight]
      %p_features_7_conv_1_1_weight : [num_users=1] = placeholder[target=p_features_7_conv_1_1_weight]
      %p_features_7_conv_1_1_bias : [num_users=1] = placeholder[target=p_features_7_conv_1_1_bias]
      %p_features_7_conv_2_weight : [num_users=1] = placeholder[target=p_features_7_conv_2_weight]
      %p_features_7_conv_3_weight : [num_users=1] = placeholder[target=p_features_7_conv_3_weight]
      %p_features_7_conv_3_bias : [num_users=1] = placeholder[target=p_features_7_conv_3_bias]
      %p_features_8_conv_0_0_weight : [num_users=1] = placeholder[target=p_features_8_conv_0_0_weight]
      %p_features_8_conv_0_1_weight : [num_users=1] = placeholder[target=p_features_8_conv_0_1_weight]
      %p_features_8_conv_0_1_bias : [num_users=1] = placeholder[target=p_features_8_conv_0_1_bias]
      %p_features_8_conv_1_0_weight : [num_users=1] = placeholder[target=p_features_8_conv_1_0_weight]
      %p_features_8_conv_1_1_weight : [num_users=1] = placeholder[target=p_features_8_conv_1_1_weight]
      %p_features_8_conv_1_1_bias : [num_users=1] = placeholder[target=p_features_8_conv_1_1_bias]
      %p_features_8_conv_2_weight : [num_users=1] = placeholder[target=p_features_8_conv_2_weight]
      %p_features_8_conv_3_weight : [num_users=1] = placeholder[target=p_features_8_conv_3_weight]
      %p_features_8_conv_3_bias : [num_users=1] = placeholder[target=p_features_8_conv_3_bias]
      %p_features_9_conv_0_0_weight : [num_users=1] = placeholder[target=p_features_9_conv_0_0_weight]
      %p_features_9_conv_0_1_weight : [num_users=1] = placeholder[target=p_features_9_conv_0_1_weight]
      %p_features_9_conv_0_1_bias : [num_users=1] = placeholder[target=p_features_9_conv_0_1_bias]
      %p_features_9_conv_1_0_weight : [num_users=1] = placeholder[target=p_features_9_conv_1_0_weight]
      %p_features_9_conv_1_1_weight : [num_users=1] = placeholder[target=p_features_9_conv_1_1_weight]
      %p_features_9_conv_1_1_bias : [num_users=1] = placeholder[target=p_features_9_conv_1_1_bias]
      %p_features_9_conv_2_weight : [num_users=1] = placeholder[target=p_features_9_conv_2_weight]
      %p_features_9_conv_3_weight : [num_users=1] = placeholder[target=p_features_9_conv_3_weight]
      %p_features_9_conv_3_bias : [num_users=1] = placeholder[target=p_features_9_conv_3_bias]
      %p_features_10_conv_0_0_weight : [num_users=1] = placeholder[target=p_features_10_conv_0_0_weight]
      %p_features_10_conv_0_1_weight : [num_users=1] = placeholder[target=p_features_10_conv_0_1_weight]
      %p_features_10_conv_0_1_bias : [num_users=1] = placeholder[target=p_features_10_conv_0_1_bias]
      %p_features_10_conv_1_0_weight : [num_users=1] = placeholder[target=p_features_10_conv_1_0_weight]
      %p_features_10_conv_1_1_weight : [num_users=1] = placeholder[target=p_features_10_conv_1_1_weight]
      %p_features_10_conv_1_1_bias : [num_users=1] = placeholder[target=p_features_10_conv_1_1_bias]
      %p_features_10_conv_2_weight : [num_users=1] = placeholder[target=p_features_10_conv_2_weight]
      %p_features_10_conv_3_weight : [num_users=1] = placeholder[target=p_features_10_conv_3_weight]
      %p_features_10_conv_3_bias : [num_users=1] = placeholder[target=p_features_10_conv_3_bias]
      %p_features_11_conv_0_0_weight : [num_users=1] = placeholder[target=p_features_11_conv_0_0_weight]
      %p_features_11_conv_0_1_weight : [num_users=1] = placeholder[target=p_features_11_conv_0_1_weight]
      %p_features_11_conv_0_1_bias : [num_users=1] = placeholder[target=p_features_11_conv_0_1_bias]
      %p_features_11_conv_1_0_weight : [num_users=1] = placeholder[target=p_features_11_conv_1_0_weight]
      %p_features_11_conv_1_1_weight : [num_users=1] = placeholder[target=p_features_11_conv_1_1_weight]
      %p_features_11_conv_1_1_bias : [num_users=1] = placeholder[target=p_features_11_conv_1_1_bias]
      %p_features_11_conv_2_weight : [num_users=1] = placeholder[target=p_features_11_conv_2_weight]
      %p_features_11_conv_3_weight : [num_users=1] = placeholder[target=p_features_11_conv_3_weight]
      %p_features_11_conv_3_bias : [num_users=1] = placeholder[target=p_features_11_conv_3_bias]
      %p_features_12_conv_0_0_weight : [num_users=1] = placeholder[target=p_features_12_conv_0_0_weight]
      %p_features_12_conv_0_1_weight : [num_users=1] = placeholder[target=p_features_12_conv_0_1_weight]
      %p_features_12_conv_0_1_bias : [num_users=1] = placeholder[target=p_features_12_conv_0_1_bias]
      %p_features_12_conv_1_0_weight : [num_users=1] = placeholder[target=p_features_12_conv_1_0_weight]
      %p_features_12_conv_1_1_weight : [num_users=1] = placeholder[target=p_features_12_conv_1_1_weight]
      %p_features_12_conv_1_1_bias : [num_users=1] = placeholder[target=p_features_12_conv_1_1_bias]
      %p_features_12_conv_2_weight : [num_users=1] = placeholder[target=p_features_12_conv_2_weight]
      %p_features_12_conv_3_weight : [num_users=1] = placeholder[target=p_features_12_conv_3_weight]
      %p_features_12_conv_3_bias : [num_users=1] = placeholder[target=p_features_12_conv_3_bias]
      %p_features_13_conv_0_0_weight : [num_users=1] = placeholder[target=p_features_13_conv_0_0_weight]
      %p_features_13_conv_0_1_weight : [num_users=1] = placeholder[target=p_features_13_conv_0_1_weight]
      %p_features_13_conv_0_1_bias : [num_users=1] = placeholder[target=p_features_13_conv_0_1_bias]
      %p_features_13_conv_1_0_weight : [num_users=1] = placeholder[target=p_features_13_conv_1_0_weight]
      %p_features_13_conv_1_1_weight : [num_users=1] = placeholder[target=p_features_13_conv_1_1_weight]
      %p_features_13_conv_1_1_bias : [num_users=1] = placeholder[target=p_features_13_conv_1_1_bias]
      %p_features_13_conv_2_weight : [num_users=1] = placeholder[target=p_features_13_conv_2_weight]
      %p_features_13_conv_3_weight : [num_users=1] = placeholder[target=p_features_13_conv_3_weight]
      %p_features_13_conv_3_bias : [num_users=1] = placeholder[target=p_features_13_conv_3_bias]
      %p_features_14_conv_0_0_weight : [num_users=1] = placeholder[target=p_features_14_conv_0_0_weight]
      %p_features_14_conv_0_1_weight : [num_users=1] = placeholder[target=p_features_14_conv_0_1_weight]
      %p_features_14_conv_0_1_bias : [num_users=1] = placeholder[target=p_features_14_conv_0_1_bias]
      %p_features_14_conv_1_0_weight : [num_users=1] = placeholder[target=p_features_14_conv_1_0_weight]
      %p_features_14_conv_1_1_weight : [num_users=1] = placeholder[target=p_features_14_conv_1_1_weight]
      %p_features_14_conv_1_1_bias : [num_users=1] = placeholder[target=p_features_14_conv_1_1_bias]
      %p_features_14_conv_2_weight : [num_users=1] = placeholder[target=p_features_14_conv_2_weight]
      %p_features_14_conv_3_weight : [num_users=1] = placeholder[target=p_features_14_conv_3_weight]
      %p_features_14_conv_3_bias : [num_users=1] = placeholder[target=p_features_14_conv_3_bias]
      %p_features_15_conv_0_0_weight : [num_users=1] = placeholder[target=p_features_15_conv_0_0_weight]
      %p_features_15_conv_0_1_weight : [num_users=1] = placeholder[target=p_features_15_conv_0_1_weight]
      %p_features_15_conv_0_1_bias : [num_users=1] = placeholder[target=p_features_15_conv_0_1_bias]
      %p_features_15_conv_1_0_weight : [num_users=1] = placeholder[target=p_features_15_conv_1_0_weight]
      %p_features_15_conv_1_1_weight : [num_users=1] = placeholder[target=p_features_15_conv_1_1_weight]
      %p_features_15_conv_1_1_bias : [num_users=1] = placeholder[target=p_features_15_conv_1_1_bias]
      %p_features_15_conv_2_weight : [num_users=1] = placeholder[target=p_features_15_conv_2_weight]
      %p_features_15_conv_3_weight : [num_users=1] = placeholder[target=p_features_15_conv_3_weight]
      %p_features_15_conv_3_bias : [num_users=1] = placeholder[target=p_features_15_conv_3_bias]
      %p_features_16_conv_0_0_weight : [num_users=1] = placeholder[target=p_features_16_conv_0_0_weight]
      %p_features_16_conv_0_1_weight : [num_users=1] = placeholder[target=p_features_16_conv_0_1_weight]
      %p_features_16_conv_0_1_bias : [num_users=1] = placeholder[target=p_features_16_conv_0_1_bias]
      %p_features_16_conv_1_0_weight : [num_users=1] = placeholder[target=p_features_16_conv_1_0_weight]
      %p_features_16_conv_1_1_weight : [num_users=1] = placeholder[target=p_features_16_conv_1_1_weight]
      %p_features_16_conv_1_1_bias : [num_users=1] = placeholder[target=p_features_16_conv_1_1_bias]
      %p_features_16_conv_2_weight : [num_users=1] = placeholder[target=p_features_16_conv_2_weight]
      %p_features_16_conv_3_weight : [num_users=1] = placeholder[target=p_features_16_conv_3_weight]
      %p_features_16_conv_3_bias : [num_users=1] = placeholder[target=p_features_16_conv_3_bias]
      %p_features_17_conv_0_0_weight : [num_users=1] = placeholder[target=p_features_17_conv_0_0_weight]
      %p_features_17_conv_0_1_weight : [num_users=1] = placeholder[target=p_features_17_conv_0_1_weight]
      %p_features_17_conv_0_1_bias : [num_users=1] = placeholder[target=p_features_17_conv_0_1_bias]
      %p_features_17_conv_1_0_weight : [num_users=1] = placeholder[target=p_features_17_conv_1_0_weight]
      %p_features_17_conv_1_1_weight : [num_users=1] = placeholder[target=p_features_17_conv_1_1_weight]
      %p_features_17_conv_1_1_bias : [num_users=1] = placeholder[target=p_features_17_conv_1_1_bias]
      %p_features_17_conv_2_weight : [num_users=1] = placeholder[target=p_features_17_conv_2_weight]
      %p_features_17_conv_3_weight : [num_users=1] = placeholder[target=p_features_17_conv_3_weight]
      %p_features_17_conv_3_bias : [num_users=1] = placeholder[target=p_features_17_conv_3_bias]
      %p_features_18_0_weight : [num_users=1] = placeholder[target=p_features_18_0_weight]
      %p_features_18_1_weight : [num_users=1] = placeholder[target=p_features_18_1_weight]
      %p_features_18_1_bias : [num_users=1] = placeholder[target=p_features_18_1_bias]
      %b_features_0_1_running_mean : [num_users=1] = placeholder[target=b_features_0_1_running_mean]
      %b_features_0_1_running_var : [num_users=1] = placeholder[target=b_features_0_1_running_var]
      %b_features_1_conv_0_1_running_mean : [num_users=1] = placeholder[target=b_features_1_conv_0_1_running_mean]
      %b_features_1_conv_0_1_running_var : [num_users=1] = placeholder[target=b_features_1_conv_0_1_running_var]
      %b_features_1_conv_2_running_mean : [num_users=1] = placeholder[target=b_features_1_conv_2_running_mean]
      %b_features_1_conv_2_running_var : [num_users=1] = placeholder[target=b_features_1_conv_2_running_var]
      %b_features_2_conv_0_1_running_mean : [num_users=1] = placeholder[target=b_features_2_conv_0_1_running_mean]
      %b_features_2_conv_0_1_running_var : [num_users=1] = placeholder[target=b_features_2_conv_0_1_running_var]
      %b_features_2_conv_1_1_running_mean : [num_users=1] = placeholder[target=b_features_2_conv_1_1_running_mean]
      %b_features_2_conv_1_1_running_var : [num_users=1] = placeholder[target=b_features_2_conv_1_1_running_var]
      %b_features_2_conv_3_running_mean : [num_users=1] = placeholder[target=b_features_2_conv_3_running_mean]
      %b_features_2_conv_3_running_var : [num_users=1] = placeholder[target=b_features_2_conv_3_running_var]
      %b_features_3_conv_0_1_running_mean : [num_users=1] = placeholder[target=b_features_3_conv_0_1_running_mean]
      %b_features_3_conv_0_1_running_var : [num_users=1] = placeholder[target=b_features_3_conv_0_1_running_var]
      %b_features_3_conv_1_1_running_mean : [num_users=1] = placeholder[target=b_features_3_conv_1_1_running_mean]
      %b_features_3_conv_1_1_running_var : [num_users=1] = placeholder[target=b_features_3_conv_1_1_running_var]
      %b_features_3_conv_3_running_mean : [num_users=1] = placeholder[target=b_features_3_conv_3_running_mean]
      %b_features_3_conv_3_running_var : [num_users=1] = placeholder[target=b_features_3_conv_3_running_var]
      %b_features_4_conv_0_1_running_mean : [num_users=1] = placeholder[target=b_features_4_conv_0_1_running_mean]
      %b_features_4_conv_0_1_running_var : [num_users=1] = placeholder[target=b_features_4_conv_0_1_running_var]
      %b_features_4_conv_1_1_running_mean : [num_users=1] = placeholder[target=b_features_4_conv_1_1_running_mean]
      %b_features_4_conv_1_1_running_var : [num_users=1] = placeholder[target=b_features_4_conv_1_1_running_var]
      %b_features_4_conv_3_running_mean : [num_users=1] = placeholder[target=b_features_4_conv_3_running_mean]
      %b_features_4_conv_3_running_var : [num_users=1] = placeholder[target=b_features_4_conv_3_running_var]
      %b_features_5_conv_0_1_running_mean : [num_users=1] = placeholder[target=b_features_5_conv_0_1_running_mean]
      %b_features_5_conv_0_1_running_var : [num_users=1] = placeholder[target=b_features_5_conv_0_1_running_var]
      %b_features_5_conv_1_1_running_mean : [num_users=1] = placeholder[target=b_features_5_conv_1_1_running_mean]
      %b_features_5_conv_1_1_running_var : [num_users=1] = placeholder[target=b_features_5_conv_1_1_running_var]
      %b_features_5_conv_3_running_mean : [num_users=1] = placeholder[target=b_features_5_conv_3_running_mean]
      %b_features_5_conv_3_running_var : [num_users=1] = placeholder[target=b_features_5_conv_3_running_var]
      %b_features_6_conv_0_1_running_mean : [num_users=1] = placeholder[target=b_features_6_conv_0_1_running_mean]
      %b_features_6_conv_0_1_running_var : [num_users=1] = placeholder[target=b_features_6_conv_0_1_running_var]
      %b_features_6_conv_1_1_running_mean : [num_users=1] = placeholder[target=b_features_6_conv_1_1_running_mean]
      %b_features_6_conv_1_1_running_var : [num_users=1] = placeholder[target=b_features_6_conv_1_1_running_var]
      %b_features_6_conv_3_running_mean : [num_users=1] = placeholder[target=b_features_6_conv_3_running_mean]
      %b_features_6_conv_3_running_var : [num_users=1] = placeholder[target=b_features_6_conv_3_running_var]
      %b_features_7_conv_0_1_running_mean : [num_users=1] = placeholder[target=b_features_7_conv_0_1_running_mean]
      %b_features_7_conv_0_1_running_var : [num_users=1] = placeholder[target=b_features_7_conv_0_1_running_var]
      %b_features_7_conv_1_1_running_mean : [num_users=1] = placeholder[target=b_features_7_conv_1_1_running_mean]
      %b_features_7_conv_1_1_running_var : [num_users=1] = placeholder[target=b_features_7_conv_1_1_running_var]
      %b_features_7_conv_3_running_mean : [num_users=1] = placeholder[target=b_features_7_conv_3_running_mean]
      %b_features_7_conv_3_running_var : [num_users=1] = placeholder[target=b_features_7_conv_3_running_var]
      %b_features_8_conv_0_1_running_mean : [num_users=1] = placeholder[target=b_features_8_conv_0_1_running_mean]
      %b_features_8_conv_0_1_running_var : [num_users=1] = placeholder[target=b_features_8_conv_0_1_running_var]
      %b_features_8_conv_1_1_running_mean : [num_users=1] = placeholder[target=b_features_8_conv_1_1_running_mean]
      %b_features_8_conv_1_1_running_var : [num_users=1] = placeholder[target=b_features_8_conv_1_1_running_var]
      %b_features_8_conv_3_running_mean : [num_users=1] = placeholder[target=b_features_8_conv_3_running_mean]
      %b_features_8_conv_3_running_var : [num_users=1] = placeholder[target=b_features_8_conv_3_running_var]
      %b_features_9_conv_0_1_running_mean : [num_users=1] = placeholder[target=b_features_9_conv_0_1_running_mean]
      %b_features_9_conv_0_1_running_var : [num_users=1] = placeholder[target=b_features_9_conv_0_1_running_var]
      %b_features_9_conv_1_1_running_mean : [num_users=1] = placeholder[target=b_features_9_conv_1_1_running_mean]
      %b_features_9_conv_1_1_running_var : [num_users=1] = placeholder[target=b_features_9_conv_1_1_running_var]
      %b_features_9_conv_3_running_mean : [num_users=1] = placeholder[target=b_features_9_conv_3_running_mean]
      %b_features_9_conv_3_running_var : [num_users=1] = placeholder[target=b_features_9_conv_3_running_var]
      %b_features_10_conv_0_1_running_mean : [num_users=1] = placeholder[target=b_features_10_conv_0_1_running_mean]
      %b_features_10_conv_0_1_running_var : [num_users=1] = placeholder[target=b_features_10_conv_0_1_running_var]
      %b_features_10_conv_1_1_running_mean : [num_users=1] = placeholder[target=b_features_10_conv_1_1_running_mean]
      %b_features_10_conv_1_1_running_var : [num_users=1] = placeholder[target=b_features_10_conv_1_1_running_var]
      %b_features_10_conv_3_running_mean : [num_users=1] = placeholder[target=b_features_10_conv_3_running_mean]
      %b_features_10_conv_3_running_var : [num_users=1] = placeholder[target=b_features_10_conv_3_running_var]
      %b_features_11_conv_0_1_running_mean : [num_users=1] = placeholder[target=b_features_11_conv_0_1_running_mean]
      %b_features_11_conv_0_1_running_var : [num_users=1] = placeholder[target=b_features_11_conv_0_1_running_var]
      %b_features_11_conv_1_1_running_mean : [num_users=1] = placeholder[target=b_features_11_conv_1_1_running_mean]
      %b_features_11_conv_1_1_running_var : [num_users=1] = placeholder[target=b_features_11_conv_1_1_running_var]
      %b_features_11_conv_3_running_mean : [num_users=1] = placeholder[target=b_features_11_conv_3_running_mean]
      %b_features_11_conv_3_running_var : [num_users=1] = placeholder[target=b_features_11_conv_3_running_var]
      %b_features_12_conv_0_1_running_mean : [num_users=1] = placeholder[target=b_features_12_conv_0_1_running_mean]
      %b_features_12_conv_0_1_running_var : [num_users=1] = placeholder[target=b_features_12_conv_0_1_running_var]
      %b_features_12_conv_1_1_running_mean : [num_users=1] = placeholder[target=b_features_12_conv_1_1_running_mean]
      %b_features_12_conv_1_1_running_var : [num_users=1] = placeholder[target=b_features_12_conv_1_1_running_var]
      %b_features_12_conv_3_running_mean : [num_users=1] = placeholder[target=b_features_12_conv_3_running_mean]
      %b_features_12_conv_3_running_var : [num_users=1] = placeholder[target=b_features_12_conv_3_running_var]
      %b_features_13_conv_0_1_running_mean : [num_users=1] = placeholder[target=b_features_13_conv_0_1_running_mean]
      %b_features_13_conv_0_1_running_var : [num_users=1] = placeholder[target=b_features_13_conv_0_1_running_var]
      %b_features_13_conv_1_1_running_mean : [num_users=1] = placeholder[target=b_features_13_conv_1_1_running_mean]
      %b_features_13_conv_1_1_running_var : [num_users=1] = placeholder[target=b_features_13_conv_1_1_running_var]
      %b_features_13_conv_3_running_mean : [num_users=1] = placeholder[target=b_features_13_conv_3_running_mean]
      %b_features_13_conv_3_running_var : [num_users=1] = placeholder[target=b_features_13_conv_3_running_var]
      %b_features_14_conv_0_1_running_mean : [num_users=1] = placeholder[target=b_features_14_conv_0_1_running_mean]
      %b_features_14_conv_0_1_running_var : [num_users=1] = placeholder[target=b_features_14_conv_0_1_running_var]
      %b_features_14_conv_1_1_running_mean : [num_users=1] = placeholder[target=b_features_14_conv_1_1_running_mean]
      %b_features_14_conv_1_1_running_var : [num_users=1] = placeholder[target=b_features_14_conv_1_1_running_var]
      %b_features_14_conv_3_running_mean : [num_users=1] = placeholder[target=b_features_14_conv_3_running_mean]
      %b_features_14_conv_3_running_var : [num_users=1] = placeholder[target=b_features_14_conv_3_running_var]
      %b_features_15_conv_0_1_running_mean : [num_users=1] = placeholder[target=b_features_15_conv_0_1_running_mean]
      %b_features_15_conv_0_1_running_var : [num_users=1] = placeholder[target=b_features_15_conv_0_1_running_var]
      %b_features_15_conv_1_1_running_mean : [num_users=1] = placeholder[target=b_features_15_conv_1_1_running_mean]
      %b_features_15_conv_1_1_running_var : [num_users=1] = placeholder[target=b_features_15_conv_1_1_running_var]
      %b_features_15_conv_3_running_mean : [num_users=1] = placeholder[target=b_features_15_conv_3_running_mean]
      %b_features_15_conv_3_running_var : [num_users=1] = placeholder[target=b_features_15_conv_3_running_var]
      %b_features_16_conv_0_1_running_mean : [num_users=1] = placeholder[target=b_features_16_conv_0_1_running_mean]
      %b_features_16_conv_0_1_running_var : [num_users=1] = placeholder[target=b_features_16_conv_0_1_running_var]
      %b_features_16_conv_1_1_running_mean : [num_users=1] = placeholder[target=b_features_16_conv_1_1_running_mean]
      %b_features_16_conv_1_1_running_var : [num_users=1] = placeholder[target=b_features_16_conv_1_1_running_var]
      %b_features_16_conv_3_running_mean : [num_users=1] = placeholder[target=b_features_16_conv_3_running_mean]
      %b_features_16_conv_3_running_var : [num_users=1] = placeholder[target=b_features_16_conv_3_running_var]
      %b_features_17_conv_0_1_running_mean : [num_users=1] = placeholder[target=b_features_17_conv_0_1_running_mean]
      %b_features_17_conv_0_1_running_var : [num_users=1] = placeholder[target=b_features_17_conv_0_1_running_var]
      %b_features_17_conv_1_1_running_mean : [num_users=1] = placeholder[target=b_features_17_conv_1_1_running_mean]
      %b_features_17_conv_1_1_running_var : [num_users=1] = placeholder[target=b_features_17_conv_1_1_running_var]
      %b_features_17_conv_3_running_mean : [num_users=1] = placeholder[target=b_features_17_conv_3_running_mean]
      %b_features_17_conv_3_running_var : [num_users=1] = placeholder[target=b_features_17_conv_3_running_var]
      %b_features_18_1_running_mean : [num_users=1] = placeholder[target=b_features_18_1_running_mean]
      %b_features_18_1_running_var : [num_users=1] = placeholder[target=b_features_18_1_running_var]
      %x : [num_users=1] = placeholder[target=x]
      %aten_convolution_default : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%x, %p_features_0_0_weight, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default, %p_features_0_1_weight, %p_features_0_1_bias, %b_features_0_1_running_mean, %b_features_0_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default, 0), kwargs = {})
      %aten_hardtanh_default : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_1 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default, %p_features_1_conv_0_0_weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_1 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_1, %p_features_1_conv_0_1_weight, %p_features_1_conv_0_1_bias, %b_features_1_conv_0_1_running_mean, %b_features_1_conv_0_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_1, 0), kwargs = {})
      %aten_hardtanh_default_1 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_1, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_2 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_1, %p_features_1_conv_1_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_2 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_2, %p_features_1_conv_2_weight, %p_features_1_conv_2_bias, %b_features_1_conv_2_running_mean, %b_features_1_conv_2_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_2 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_2, 0), kwargs = {})
      %aten_convolution_default_3 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%getitem_2, %p_features_2_conv_0_0_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_3 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_3, %p_features_2_conv_0_1_weight, %p_features_2_conv_0_1_bias, %b_features_2_conv_0_1_running_mean, %b_features_2_conv_0_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_3, 0), kwargs = {})
      %aten_hardtanh_default_2 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_3, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_4 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_2, %p_features_2_conv_1_0_weight, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 96), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_4 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_4, %p_features_2_conv_1_1_weight, %p_features_2_conv_1_1_bias, %b_features_2_conv_1_1_running_mean, %b_features_2_conv_1_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_4 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_4, 0), kwargs = {})
      %aten_hardtanh_default_3 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_4, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_5 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_3, %p_features_2_conv_2_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_5 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_5, %p_features_2_conv_3_weight, %p_features_2_conv_3_bias, %b_features_2_conv_3_running_mean, %b_features_2_conv_3_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_5 : [num_users=2] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_5, 0), kwargs = {})
      %aten_convolution_default_6 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%getitem_5, %p_features_3_conv_0_0_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_6 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_6, %p_features_3_conv_0_1_weight, %p_features_3_conv_0_1_bias, %b_features_3_conv_0_1_running_mean, %b_features_3_conv_0_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_6 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_6, 0), kwargs = {})
      %aten_hardtanh_default_4 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_6, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_7 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_4, %p_features_3_conv_1_0_weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 144), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_7 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_7, %p_features_3_conv_1_1_weight, %p_features_3_conv_1_1_bias, %b_features_3_conv_1_1_running_mean, %b_features_3_conv_1_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_7 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_7, 0), kwargs = {})
      %aten_hardtanh_default_5 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_7, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_8 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_5, %p_features_3_conv_2_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_8 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_8, %p_features_3_conv_3_weight, %p_features_3_conv_3_bias, %b_features_3_conv_3_running_mean, %b_features_3_conv_3_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_8 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_8, 0), kwargs = {})
      %aten_add_tensor : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%getitem_5, %getitem_8), kwargs = {})
      %aten_convolution_default_9 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_add_tensor, %p_features_4_conv_0_0_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_9 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_9, %p_features_4_conv_0_1_weight, %p_features_4_conv_0_1_bias, %b_features_4_conv_0_1_running_mean, %b_features_4_conv_0_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_9 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_9, 0), kwargs = {})
      %aten_hardtanh_default_6 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_9, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_10 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_6, %p_features_4_conv_1_0_weight, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 144), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_10 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_10, %p_features_4_conv_1_1_weight, %p_features_4_conv_1_1_bias, %b_features_4_conv_1_1_running_mean, %b_features_4_conv_1_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_10 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_10, 0), kwargs = {})
      %aten_hardtanh_default_7 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_10, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_11 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_7, %p_features_4_conv_2_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_11 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_11, %p_features_4_conv_3_weight, %p_features_4_conv_3_bias, %b_features_4_conv_3_running_mean, %b_features_4_conv_3_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_11 : [num_users=2] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_11, 0), kwargs = {})
      %aten_convolution_default_12 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%getitem_11, %p_features_5_conv_0_0_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_12 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_12, %p_features_5_conv_0_1_weight, %p_features_5_conv_0_1_bias, %b_features_5_conv_0_1_running_mean, %b_features_5_conv_0_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_12 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_12, 0), kwargs = {})
      %aten_hardtanh_default_8 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_12, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_13 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_8, %p_features_5_conv_1_0_weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 192), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_13 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_13, %p_features_5_conv_1_1_weight, %p_features_5_conv_1_1_bias, %b_features_5_conv_1_1_running_mean, %b_features_5_conv_1_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_13 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_13, 0), kwargs = {})
      %aten_hardtanh_default_9 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_13, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_14 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_9, %p_features_5_conv_2_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_14 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_14, %p_features_5_conv_3_weight, %p_features_5_conv_3_bias, %b_features_5_conv_3_running_mean, %b_features_5_conv_3_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_14 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_14, 0), kwargs = {})
      %aten_add_tensor_1 : [num_users=2] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%getitem_11, %getitem_14), kwargs = {})
      %aten_convolution_default_15 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_add_tensor_1, %p_features_6_conv_0_0_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_15 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_15, %p_features_6_conv_0_1_weight, %p_features_6_conv_0_1_bias, %b_features_6_conv_0_1_running_mean, %b_features_6_conv_0_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_15 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_15, 0), kwargs = {})
      %aten_hardtanh_default_10 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_15, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_16 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_10, %p_features_6_conv_1_0_weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 192), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_16 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_16, %p_features_6_conv_1_1_weight, %p_features_6_conv_1_1_bias, %b_features_6_conv_1_1_running_mean, %b_features_6_conv_1_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_16 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_16, 0), kwargs = {})
      %aten_hardtanh_default_11 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_16, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_17 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_11, %p_features_6_conv_2_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_17 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_17, %p_features_6_conv_3_weight, %p_features_6_conv_3_bias, %b_features_6_conv_3_running_mean, %b_features_6_conv_3_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_17 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_17, 0), kwargs = {})
      %aten_add_tensor_2 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%aten_add_tensor_1, %getitem_17), kwargs = {})
      %aten_convolution_default_18 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_add_tensor_2, %p_features_7_conv_0_0_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_18 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_18, %p_features_7_conv_0_1_weight, %p_features_7_conv_0_1_bias, %b_features_7_conv_0_1_running_mean, %b_features_7_conv_0_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_18 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_18, 0), kwargs = {})
      %aten_hardtanh_default_12 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_18, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_19 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_12, %p_features_7_conv_1_0_weight, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 192), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_19 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_19, %p_features_7_conv_1_1_weight, %p_features_7_conv_1_1_bias, %b_features_7_conv_1_1_running_mean, %b_features_7_conv_1_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_19 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_19, 0), kwargs = {})
      %aten_hardtanh_default_13 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_19, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_20 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_13, %p_features_7_conv_2_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_20 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_20, %p_features_7_conv_3_weight, %p_features_7_conv_3_bias, %b_features_7_conv_3_running_mean, %b_features_7_conv_3_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_20 : [num_users=2] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_20, 0), kwargs = {})
      %aten_convolution_default_21 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%getitem_20, %p_features_8_conv_0_0_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_21 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_21, %p_features_8_conv_0_1_weight, %p_features_8_conv_0_1_bias, %b_features_8_conv_0_1_running_mean, %b_features_8_conv_0_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_21 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_21, 0), kwargs = {})
      %aten_hardtanh_default_14 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_21, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_22 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_14, %p_features_8_conv_1_0_weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 384), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_22 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_22, %p_features_8_conv_1_1_weight, %p_features_8_conv_1_1_bias, %b_features_8_conv_1_1_running_mean, %b_features_8_conv_1_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_22 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_22, 0), kwargs = {})
      %aten_hardtanh_default_15 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_22, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_23 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_15, %p_features_8_conv_2_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_23 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_23, %p_features_8_conv_3_weight, %p_features_8_conv_3_bias, %b_features_8_conv_3_running_mean, %b_features_8_conv_3_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_23 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_23, 0), kwargs = {})
      %aten_add_tensor_3 : [num_users=2] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%getitem_20, %getitem_23), kwargs = {})
      %aten_convolution_default_24 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_add_tensor_3, %p_features_9_conv_0_0_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_24 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_24, %p_features_9_conv_0_1_weight, %p_features_9_conv_0_1_bias, %b_features_9_conv_0_1_running_mean, %b_features_9_conv_0_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_24 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_24, 0), kwargs = {})
      %aten_hardtanh_default_16 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_24, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_25 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_16, %p_features_9_conv_1_0_weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 384), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_25 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_25, %p_features_9_conv_1_1_weight, %p_features_9_conv_1_1_bias, %b_features_9_conv_1_1_running_mean, %b_features_9_conv_1_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_25 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_25, 0), kwargs = {})
      %aten_hardtanh_default_17 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_25, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_26 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_17, %p_features_9_conv_2_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_26 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_26, %p_features_9_conv_3_weight, %p_features_9_conv_3_bias, %b_features_9_conv_3_running_mean, %b_features_9_conv_3_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_26 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_26, 0), kwargs = {})
      %aten_add_tensor_4 : [num_users=2] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%aten_add_tensor_3, %getitem_26), kwargs = {})
      %aten_convolution_default_27 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_add_tensor_4, %p_features_10_conv_0_0_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_27 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_27, %p_features_10_conv_0_1_weight, %p_features_10_conv_0_1_bias, %b_features_10_conv_0_1_running_mean, %b_features_10_conv_0_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_27 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_27, 0), kwargs = {})
      %aten_hardtanh_default_18 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_27, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_28 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_18, %p_features_10_conv_1_0_weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 384), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_28 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_28, %p_features_10_conv_1_1_weight, %p_features_10_conv_1_1_bias, %b_features_10_conv_1_1_running_mean, %b_features_10_conv_1_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_28 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_28, 0), kwargs = {})
      %aten_hardtanh_default_19 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_28, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_29 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_19, %p_features_10_conv_2_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_29 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_29, %p_features_10_conv_3_weight, %p_features_10_conv_3_bias, %b_features_10_conv_3_running_mean, %b_features_10_conv_3_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_29 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_29, 0), kwargs = {})
      %aten_add_tensor_5 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%aten_add_tensor_4, %getitem_29), kwargs = {})
      %aten_convolution_default_30 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_add_tensor_5, %p_features_11_conv_0_0_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_30 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_30, %p_features_11_conv_0_1_weight, %p_features_11_conv_0_1_bias, %b_features_11_conv_0_1_running_mean, %b_features_11_conv_0_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_30 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_30, 0), kwargs = {})
      %aten_hardtanh_default_20 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_30, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_31 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_20, %p_features_11_conv_1_0_weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 384), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_31 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_31, %p_features_11_conv_1_1_weight, %p_features_11_conv_1_1_bias, %b_features_11_conv_1_1_running_mean, %b_features_11_conv_1_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_31 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_31, 0), kwargs = {})
      %aten_hardtanh_default_21 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_31, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_32 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_21, %p_features_11_conv_2_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_32 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_32, %p_features_11_conv_3_weight, %p_features_11_conv_3_bias, %b_features_11_conv_3_running_mean, %b_features_11_conv_3_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_32 : [num_users=2] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_32, 0), kwargs = {})
      %aten_convolution_default_33 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%getitem_32, %p_features_12_conv_0_0_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_33 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_33, %p_features_12_conv_0_1_weight, %p_features_12_conv_0_1_bias, %b_features_12_conv_0_1_running_mean, %b_features_12_conv_0_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_33 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_33, 0), kwargs = {})
      %aten_hardtanh_default_22 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_33, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_34 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_22, %p_features_12_conv_1_0_weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 576), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_34 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_34, %p_features_12_conv_1_1_weight, %p_features_12_conv_1_1_bias, %b_features_12_conv_1_1_running_mean, %b_features_12_conv_1_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_34 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_34, 0), kwargs = {})
      %aten_hardtanh_default_23 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_34, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_35 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_23, %p_features_12_conv_2_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_35 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_35, %p_features_12_conv_3_weight, %p_features_12_conv_3_bias, %b_features_12_conv_3_running_mean, %b_features_12_conv_3_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_35 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_35, 0), kwargs = {})
      %aten_add_tensor_6 : [num_users=2] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%getitem_32, %getitem_35), kwargs = {})
      %aten_convolution_default_36 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_add_tensor_6, %p_features_13_conv_0_0_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_36 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_36, %p_features_13_conv_0_1_weight, %p_features_13_conv_0_1_bias, %b_features_13_conv_0_1_running_mean, %b_features_13_conv_0_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_36 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_36, 0), kwargs = {})
      %aten_hardtanh_default_24 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_36, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_37 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_24, %p_features_13_conv_1_0_weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 576), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_37 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_37, %p_features_13_conv_1_1_weight, %p_features_13_conv_1_1_bias, %b_features_13_conv_1_1_running_mean, %b_features_13_conv_1_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_37 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_37, 0), kwargs = {})
      %aten_hardtanh_default_25 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_37, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_38 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_25, %p_features_13_conv_2_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_38 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_38, %p_features_13_conv_3_weight, %p_features_13_conv_3_bias, %b_features_13_conv_3_running_mean, %b_features_13_conv_3_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_38 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_38, 0), kwargs = {})
      %aten_add_tensor_7 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%aten_add_tensor_6, %getitem_38), kwargs = {})
      %aten_convolution_default_39 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_add_tensor_7, %p_features_14_conv_0_0_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_39 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_39, %p_features_14_conv_0_1_weight, %p_features_14_conv_0_1_bias, %b_features_14_conv_0_1_running_mean, %b_features_14_conv_0_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_39 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_39, 0), kwargs = {})
      %aten_hardtanh_default_26 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_39, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_40 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_26, %p_features_14_conv_1_0_weight, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 576), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_40 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_40, %p_features_14_conv_1_1_weight, %p_features_14_conv_1_1_bias, %b_features_14_conv_1_1_running_mean, %b_features_14_conv_1_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_40 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_40, 0), kwargs = {})
      %aten_hardtanh_default_27 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_40, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_41 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_27, %p_features_14_conv_2_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_41 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_41, %p_features_14_conv_3_weight, %p_features_14_conv_3_bias, %b_features_14_conv_3_running_mean, %b_features_14_conv_3_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_41 : [num_users=2] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_41, 0), kwargs = {})
      %aten_convolution_default_42 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%getitem_41, %p_features_15_conv_0_0_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_42 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_42, %p_features_15_conv_0_1_weight, %p_features_15_conv_0_1_bias, %b_features_15_conv_0_1_running_mean, %b_features_15_conv_0_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_42 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_42, 0), kwargs = {})
      %aten_hardtanh_default_28 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_42, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_43 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_28, %p_features_15_conv_1_0_weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 960), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_43 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_43, %p_features_15_conv_1_1_weight, %p_features_15_conv_1_1_bias, %b_features_15_conv_1_1_running_mean, %b_features_15_conv_1_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_43 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_43, 0), kwargs = {})
      %aten_hardtanh_default_29 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_43, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_44 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_29, %p_features_15_conv_2_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_44 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_44, %p_features_15_conv_3_weight, %p_features_15_conv_3_bias, %b_features_15_conv_3_running_mean, %b_features_15_conv_3_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_44 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_44, 0), kwargs = {})
      %aten_add_tensor_8 : [num_users=2] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%getitem_41, %getitem_44), kwargs = {})
      %aten_convolution_default_45 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_add_tensor_8, %p_features_16_conv_0_0_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_45 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_45, %p_features_16_conv_0_1_weight, %p_features_16_conv_0_1_bias, %b_features_16_conv_0_1_running_mean, %b_features_16_conv_0_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_45 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_45, 0), kwargs = {})
      %aten_hardtanh_default_30 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_45, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_46 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_30, %p_features_16_conv_1_0_weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 960), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_46 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_46, %p_features_16_conv_1_1_weight, %p_features_16_conv_1_1_bias, %b_features_16_conv_1_1_running_mean, %b_features_16_conv_1_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_46 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_46, 0), kwargs = {})
      %aten_hardtanh_default_31 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_46, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_47 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_31, %p_features_16_conv_2_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_47 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_47, %p_features_16_conv_3_weight, %p_features_16_conv_3_bias, %b_features_16_conv_3_running_mean, %b_features_16_conv_3_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_47 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_47, 0), kwargs = {})
      %aten_add_tensor_9 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.add.Tensor](args = (%aten_add_tensor_8, %getitem_47), kwargs = {})
      %aten_convolution_default_48 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_add_tensor_9, %p_features_17_conv_0_0_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_48 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_48, %p_features_17_conv_0_1_weight, %p_features_17_conv_0_1_bias, %b_features_17_conv_0_1_running_mean, %b_features_17_conv_0_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_48 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_48, 0), kwargs = {})
      %aten_hardtanh_default_32 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_48, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_49 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_32, %p_features_17_conv_1_0_weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 960), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_49 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_49, %p_features_17_conv_1_1_weight, %p_features_17_conv_1_1_bias, %b_features_17_conv_1_1_running_mean, %b_features_17_conv_1_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_49 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_49, 0), kwargs = {})
      %aten_hardtanh_default_33 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_49, 0.0, 6.0), kwargs = {})
      %aten_convolution_default_50 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%aten_hardtanh_default_33, %p_features_17_conv_2_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_50 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_50, %p_features_17_conv_3_weight, %p_features_17_conv_3_bias, %b_features_17_conv_3_running_mean, %b_features_17_conv_3_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_50 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_50, 0), kwargs = {})
      %aten_convolution_default_51 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.convolution.default](args = (%getitem_50, %p_features_18_0_weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
      %aten__native_batch_norm_legit_no_training_default_51 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten._native_batch_norm_legit_no_training.default](args = (%aten_convolution_default_51, %p_features_18_1_weight, %p_features_18_1_bias, %b_features_18_1_running_mean, %b_features_18_1_running_var, 0.1, 1e-05), kwargs = {})
      %getitem_51 : [num_users=1] = call_function[target=operator.getitem](args = (%aten__native_batch_norm_legit_no_training_default_51, 0), kwargs = {})
      %aten_hardtanh_default_34 : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.hardtanh.default](args = (%getitem_51, 0.0, 6.0), kwargs = {})
      %aten_mean_dim : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.mean.dim](args = (%aten_hardtanh_default_34, [-1, -2], True), kwargs = {})
      return (aten_mean_dim,)
  %executorch_call_delegate : [num_users=1] = call_function[target=torch.ops.higher_order.executorch_call_delegate](args = (%lowered_module_0, %x), kwargs = {})
  %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%executorch_call_delegate, 0), kwargs = {})
  %aten_view_copy_default : [num_users=1] = call_function[target=executorch.exir.memory.view](args = (%getitem, [1, 1280]), kwargs = {})
  %alloc : [num_users=1] = call_function[target=executorch.exir.memory.alloc](args = (((1, 1280), torch.float32),), kwargs = {})
  %dim_order_ops__clone_dim_order_default : [num_users=1] = call_function[target=torch.ops.dim_order_ops._clone_dim_order.out](args = (%aten_view_copy_default,), kwargs = {dim_order: [0, 1], out: %alloc})
  %lowered_module_1 : [num_users=1] = get_attr[target=lowered_module_1]
    backend_id: XnnpackBackend
    lowered graph():
      %p_classifier_1_weight : [num_users=1] = placeholder[target=p_classifier_1_weight]
      %p_classifier_1_bias : [num_users=1] = placeholder[target=p_classifier_1_bias]
      %dim_order_ops__clone_dim_order_default : [num_users=1] = placeholder[target=dim_order_ops__clone_dim_order_default]
      %aten_linear_default : [num_users=1] = call_function[target=executorch.exir.dialects.edge._ops.aten.linear.default](args = (%dim_order_ops__clone_dim_order_default, %p_classifier_1_weight, %p_classifier_1_bias), kwargs = {})
      return (aten_linear_default,)
  %executorch_call_delegate_1 : [num_users=1] = call_function[target=torch.ops.higher_order.executorch_call_delegate](args = (%lowered_module_1, %dim_order_ops__clone_dim_order_default), kwargs = {})
  %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%executorch_call_delegate_1, 0), kwargs = {})
  return (getitem_1,)
```
</details>
