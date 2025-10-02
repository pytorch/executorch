# Visualize a Model using ModelExplorer

The [visualization_utils.py](/executorch/devtools/visualization/visualization_utils.py) contains functions for
visualizing ExecuTorch models as computational graphs using the `ModelExplorer` utility.

## Installation

To install the `ModelExplorer` and its dependencies, run:

```
./devtools/install_requirements.sh
```

## Visualize a model

The function `visualize()` takes an `ExportedProgram` and launches a `ModelExplorer` server instance. A browser tab will
open, containing the visualization.

---

# Visualize a Model with Highlighted QDQ Clusters and Partitions

The [visualization_utils.py](../../devtools/visualization/visualization_utils.py) contains the function
`visualize_with_clusters()` which takes an `ExportedProgram` and visualizes it using the `ModelExplorer` utility.  
It groups QDQ clusters and individual partitions together to improve readability. Example usage is available
in [examples/nxp/aot_neutron_compile.py](../../examples/nxp/aot_neutron_compile.py).

An example of the visualization:
![](/executorch/docs/source/_static/img/visualize_with_clusters_example.png)

## Usage

There are two main use cases for the visualization:

### 1. Launching the `ModelExplorer` and Visualizing the Model Immediately

Call:

```python
visualize_with_clusters(exported_program)
```

This starts a `ModelExplorer` server and opens a browser tab with the visualization.

By default, each call starts a new server instance and opens a new browser tab.  
To reuse an existing server, set the `reuse_server` parameter to `True`.

Starting the server is **blocking**, so the rest of your script will not run.

### 2. Storing a Serialized Graph and Visualizing Later (Non-blocking)

To save the visualization to a JSON file, call:

```python
visualize_with_clusters(exported_program, "my_model.json")
```

This just saves the visualization in the file, and it does **not** start the `ModelExplorer` server. You can then open
the file in the `ModelExplorer` GUI at any point. To launch the server, run:

```bash
  model-explorer [model-file-json]
```

If the `model-file-json` is provided, the `ModelExplorer` will open the model visualization. Otherwise, the
`ModelBuilder` GUI home page will appear. In that case, click **Select from your computer**, choose the JSON file,
and then click **View selected models** to display the graph.

---

## Styling the Graph

`visualize_with_clusters()` supports custom grouping of nodes into QDQ clusters and partitions.

You can pass the following optional parameters:

- `get_node_partition_name`
- `get_node_qdq_cluster_name`

These are functions that take a node and return a string identifying the partition or cluster it belongs to.  
Nodes with the same partition/cluster string will be grouped together and labeled accordingly in the visualization.

### Load a predefined style for QDQ cluster and partition highlighting.

A color style for the QDQ cluster and partition highlighting is already provided
in [devtools/visualization/model_explorer_styles/cluster_highlight_style.json](../../devtools/visualization/model_explorer_styles/cluster_highlight_style.json).
To load it follow these steps:

1. Click the **palette icon** in the top-right corner of the `ModelExplorer` interface.
2. Click **Import rules**.
3. Select
   the [cluster_highlight_style.json](../../devtools/visualization/model_explorer_styles/cluster_highlight_style.json)
   file to apply predefined styles that highlight each partition in a different color.
