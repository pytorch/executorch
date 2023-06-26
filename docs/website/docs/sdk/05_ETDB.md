 # Debugging with ETDB

ETDB (Executorch Debugger) is an interactive text-based tool used for investigating models (`ETRecord`) and profiling information (`ETDump`). Features provided include:
- **Tabular Visualization** of model graphs
- **Drill-in Selection** of components for detailed investigation
- **Aggregated Operator Statistics** based on results from ETDump
- **Module Architecture** grouped by a node's immediate SubModule

To kick off an ETDB instance follow these steps:


1. Generate an etrecord with the exported graph modules of your model that you want to visualize. Refer to the [etrecord tutorial](./01_generating_etrecord.md) for more details on how to do this.

2. *Optionally* generate an etdump to also visualize the performance data of your model run along with the graph. Refer to the [etdump tutorial](./02_generating_etdump.md) for more details on how to do this.

3. Initiate the ETDB instance by calling a `debug_etrecord` API, passing in the ETrecord you generated in step 1 and also the optional ETdump you generated in step 2. This will kick off the text tool. See [library documentation](./03_using_sdk_cli_tools.md) for API details

For a complete end-to-end tutorial on how to do this please refer to this [notebook](https://www.internalfb.com/intern/anp/view/?id=3799219).

---

## ETDB Flow

There are 3 stages to the ETDB flow:
1. **Graph Selection**: Prompts for which model graph to investigate
2. **Graph Overview**: Prints an overview of the selected graph
3. **Graph Investigation**: Interactive mode; User input is used to generate a focused view of the graph


### Graph Selection

Given a list of the graphs found in ETRecord, choose one to investigate.

Note: In the future it will be possible to toggle between graphs during the Graph Investigation stage.

### Graph Overview

Once a graph is selected, a set of overview tables will be displayed. Specifically the following:
- **Graph Counts**: Graph input + output counts, and a breakdown of node types in the model
- **Enumerated Nodes**: Tables listing each of the top level model nodes
- **Aggregated Summaries**: An aggregated summary table grouped by op/module type

If profiling information was attached to this graph, a **Run Summary** table with aggregated run stats will also be shown.


### Graph Investigation

Graph Investigation is the interactive segment of ETDB. With reference to the row entries in **Graph Overview**, ETDB takes as input any of the following:
- **Node Name** (e.g. add\_tensor_)
- **Module Name** (e.g. l_\_self___model_backbone_encoder_module)
- **Op Type** (e.g. add.Tensor)
- **Module Type** (e.g. Linear)

From the selection, a new set of tables containing related information will be generated.

---

#### Selection: Node Name

If an node name is provided, a few things are returned:
- **Related Nodes**: Tables containing the inputs/outputs of the selected Node
- **Parent Module Summary**: Basic information about this node's parent module (if exists)
- **Operator Summary**: Count (and summary stats if ETDump is provided) of this node's type

For example, if ETDump was provided and `addmm_default_` was selected:
- Related Nodes: Tables of Input Nodes, Tables of Output Nodes
- Parent Module Summary: Summary of `l__self___linear`
- Operator Summary: # of `addmm.default` instances in this model and runtime stats of those instances (e.g. Avg, P10, P90)

---

#### Selection: Module Name

If an module name is provided, a few things are returned:
- **Related Nodes**: Tables containing the inputs/outputs of the selected module
- **Module Contents**: Table listing all nodes within this module instance
- **Parent Module Summary**: Basic information about this node's parent module (if exists)
- **Module Summary**: Count (and summary stats if ETDump is provided) of this module's type

For example, if ETDump was provided and `l__self___linear` was selected:
- Related Nodes/Modules: Tables of Input Nodes, Tables of Output Nodes
- Module Contents: List of all nodes in `l__self___linear`
- Parent Module Summary: Summary of `l__self___wrapped_linear`
- Module Summary: # of instances of `Linear`

---

#### Selection: Op Type + Module Type

If either an op type or module type are provided, two tables are returned:
- **Aggregated Stats**: Count (and summary stats if ETDump is provided) for that node type
- **Enumerated Nodes**: Table listing all nodes of the selected type

For example, if ETDump was provided and `slice_copy.Tensor_out` was selected:
- Aggregated Stats: # of `slice_copy.Tensor_out` instances and runtime stats of those instances (e.g. Avg, P10, P90)
- Enumerated Nodes: List of all `slice_copy.Tensor_out` instances in this model
---
## Ease of Use

Below are a few ease of use features
- **Undo Investigation**: Entering `b` or `back` while in the Graph Investigation flow, replays the last user input provided
