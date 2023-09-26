# ExecuTorch XNNPACK Delegate

This subtree contains the XNNPACK Delegate implementation for ExecuTorch.
XNNPACK is an optimized library of neural network inference operators for ARM
and x86 CPUs. It is an open source project used by PyTorch. The delegate is the
mechanism for leveraging the XNNPACK library to accelerate operators running on
CPU.

## Layout
- `runtime/` : Runtime logic used at inference. This contains all the cpp files
  used to build the runtime graph and execute the XNNPACK model
- `partition/`: Partitioner is used to identify operators in model's graph that
  are suitable for lowering to XNNPACK delegate
    - `xnnpack_partitioner.py`: Contains partitioner that tags graph patterns
      for XNNPACK lowering
    - `configs.py`: Contains lists of op/modules for XNNPACK lowering
- `passes/`: Contains passes which are used before preprocessing to prepare the
  graph for XNNPACK lowering
- `operators`: the directory to store all of op visitors
    - `node_visitor.py`: Implementation of serializing each lowerable operator
      node
    - ...
- `serialization/`: Contains files related to serializing the XNNPACK graph
  representation of the PyTorch model
    - `schema.fbs`: Flatbuffer schema of serialization format
    - `xnnpack_graph_schema.py`: Python dataclasses mirroring the flatbuffer
      schema
    - `xnnpack_graph_serialize`: Implementation for serializing dataclasses
      from graph schema to flatbuffer
- `test/`: Tests for XNNPACK Delegate
- `xnnpack_preprocess.py`: Contains preprocess implementation which is called
  by `to_backend` on the graph or subgraph of a model returning a preprocessed
  blob responsible for executing the graph or subgraph at runtime

## Help & Improvements
If you have problems or questions, or have suggestions for ways to make
implementation and testing better, please reach out to the PyTorch Edge team or
create an issue on [github](https://www.github.com/pytorch/executorch/issues).

## Contributing

Please follow the following steps and guidelines when adding a new operator
implementation to this library. The goals of these guidelines are to
- Make it straightforward to add new XNNPACK operators.
- Ensure that the newly added operators are of high quality, and are easy to
  maintain
- Make it easy for users to find available operator implementations, and to
  trust in their quality and behavioral stability.

### AoT and Serialization Overview
#### Serialization:
XNNPACK delegate uses flatbuffer to serialize its nodes and values. In order to
add
[preprocessing](https://github.com/pytorch/executorch/blob/main/backends/xnnpack/xnnpack_preprocess.py)
support for a new operator, we must add the operator in both the flatbuffer
[schema](https://github.com/pytorch/executorch/blob/main/backends/xnnpack/serialization/schema.fbs),
as well as the mirrored python [data
class](https://github.com/pytorch/executorch/blob/main/backends/xnnpack/serialization/xnnpack_graph_schema.py).
These tables are based on the arguments to the XNNPACK Subgraph APIs. These
APIs can be found
[here](https://github.com/google/xnnpack/blob/master/include/xnnpack.h). We
essentially serialize all the static arguments we need to call `define_{new
operator}()`.

#### AoT Preprocess:
To add logic to preprocess new operators for the XNNPACK Delegate, we can
create new node_visitors that perform the serialization of the new operator. An
example can be found [here](). The function of these node_visitors is to
serialize all the data we define to need in the schema above.

#### AoT Partitioner:
XnnpackPartitioner is used to select the pattern (like the linear module
graph) in a big graph such that the selected nodes will be delegated to
XNNPACK. To support a new op (for example, sigmoid), add the corresponding op
or module to the
[config.py](https://github.com/pytorch/executorch/blob/main/backends/xnnpack/partition/configs.py),
which captures the sigmoid op.

#### How does it work?
- Tag the nodes: in the XNNPACK partitioner's config, which lists all ops that
  are supported by the current XNNPACK backend in executorch. When call
  `XnnpackPartitioner.partition()`, it will tag all the nodes that matches the
  patterns listed in self.pattern
- Lower the nodes; when we call `to_backend(graph_module, XnnpackPartitioner)`,
  it will loop through all the tagged nodes, and lower the group with the same
  tag.


#### Adding Tests for newly minted operators
To test newly added operators, we can add unit tests in:
[tests](https://github.com/pytorch/executorch/tree/main/backends/xnnpack/test)
