# Executorch XNNPACK Delegate

This subtree contains the XNNPACK Delegate implementation for Executorch. XNNPACK is an optimized library of neural network inference operators for ARM and x86 platforms. It is an open source projected used by PyTorch. The delegate is the mechanism for leveraging the XNNPACK Library to accelerate operators running on CPU.

## Layout
- `runtime/` : Runtime logic use at inference. This contains all the cpp files used to build the runtime graph and execute the XNNPACK model
- `partition/`: Partitioner is used to identify operators in model's graph that are suitable for lowering to XNNPACK delegate
    - `support_patterns.py`: Contains list of captured graph patterns that are suitable for XNNPack
    - `xnnpack_partitioner.py`: Contains partitioner that tags graph patterns for XNNPACK lowering
- `passes/`: Contains passes which are used before preprocessing to prepare the graph for XNNPACK lowering
- `operators`: the directory to store all of op visitors
    - `node_visitor.py`: Implementation of serializing each lowerable operator node
    - ...
- `serialization/`: Contains files related to serializing the XNNPACK graph representation of the PyTorch model
    - `schema.fbs`: Flatbuffer schema of serialization format
    - `xnnpack_graph_schema.py`: Python dataclasses mirroring the flatbuffer schema
    - `xnnpack_graph_serialize`: Implementation for serializing dataclasses from graph schema to flatbuffer
- `test/`: Tests for XNNPACK Delegate
    - `test_xnnpack.py`: end-to-end tests operator implementation of the xnnpack delegate
    - `test_xnnpack_passes.py`: Tests for graph passes used by xnnpack
- `xnnpack_preprocess.py`: Contains preprocess implementation which is called by `to_backend` on the graph or subgraph of a model returning a preprocessed blob responsible for executing the graph or subgraph at runtime

## Help & Improvements
If you have problems or questions, or have suggestions for ways to make implementation and testing better, please contact [Max Ren](https://fb.workplace.com/profile.php?id=100045762936437), [Digant Desai](https://fb.workplace.com/profile.php?id=100068306324819), or [Kimish Patel](https://fb.workplace.com/profile.php?id=100030094785558) on the PyTorch Edge team.

## Contributing

Please follow the following these steps and guidelines when adding a new operator implementation to this library. The goals of these guidelines are to
- Make it straightforward to add new XNNPack operators.
- Ensure that the newly added operators are of high quality, and are easy to maintain
- Make it easy for users to find available available operator implementations, and to trust in their quality and behavioral stability.

### AoT and Serialization Overview
#### Serialization:
XNNPACK delegate uses flatbuffer to serialize its nodes and values. In order to add [preprocessing](https://www.internalfb.com/code/fbsource/[d9018f0841600b95256187b9a08aeab2aa8b3c11]/fbcode/executorch/backends/xnnpack/xnnpack_preprocess.py?lines=357) support for a new operator, we must add the operator in both the flatbuffer [schema](https://www.internalfb.com/code/fbsource/[9a71ca4ec2a5284867562112946ac61f5660b881]/fbcode/executorch/backends/xnnpack/serialization/schema.fbs), as well as the mirrored python [data class](https://www.internalfb.com/code/fbsource/[9a71ca4ec2a5284867562112946ac61f5660b881]/fbcode/executorch/backends/xnnpack/serialization/xnnpack_graph_schema.py). These tables are based on the arguments to the XNNPACK Subgraph APIs. These APIs can be found [here](https://www.internalfb.com/code/fbsource/[9a71ca4ec2a5284867562112946ac61f5660b881]/fbcode/xplat/third-party/XNNPACK/XNNPACK/include/xnnpack.h?lines=722-729). We essentially serialize all the static arguments we need to call `define_{new operator}()`.

#### AoT Preprocess:
To add logic to preprocess new operators for the XNNPACK Delegate, we can create new node_visitors that perform the serialization of the new operator. An example can be found [here](https://www.internalfb.com/code/fbsource/[d9018f0841600b95256187b9a08aeab2aa8b3c11]/fbcode/executorch/backends/xnnpack/serialization/node_visitor.py?lines=286-314). The function of these node_visitors is to serialize all the data we define to need in the schema above.

#### AoT Partitioner:
Xnnpack Partitioner is used to selected the pattern (like the linear module graph) in a big graph such that the selected nodes will be delegated to xnnpack. To support a new op (for example, sigmoid), add the corresponding pattern to [partition/support_pattern.py](https://www.internalfb.com/code/fbsource/fbcode/executorch/backends/xnnpack/partition/support_patterns.py?lines=121-130), which captures the sigmoid op. Then expand the [self.pattern in xnnpack_partitioner.py](https://www.internalfb.com/code/fbsource/[8a7869f9d150dd6272b56d04e2d65029a92a1550]/fbcode/executorch/backends/xnnpack/partition/xnnpack_partitioner.py?lines=23-25) with the new pattern.

#### How does it work?
- Tag the nodes: in the xnnpack partitioner, there is a field called [self.patterns](https://www.internalfb.com/code/fbsource/[50683ef7e3e9baf61e1d7719e19990db3a26bbfe]/fbcode/executorch/backends/xnnpack/partition/xnnpack_partitioner.py?lines=21-29)(), which lists all ops that are supported by the current xnnpack backend in executorch. When call [xnnpackpartitioner.partition()](https://www.internalfb.com/code/fbsource/[50683ef7e3e9baf61e1d7719e19990db3a26bbfe]/fbcode/executorch/backends/xnnpack/partition/xnnpack_partitioner.py?lines=42), it will tag all the nodes that matches the patterns listed in self.pattern
- Lower the nodes; when we call `to_backend(graph_module, XnnpackPartitioner)`, it will loop through all the tagged nodes, and lower the group with the same tag.


#### Adding Tests for newly minted operators
To test newly added operators, we can add unit tests in: [test_xnnpack.py](https://www.internalfb.com/code/fbsource/[d9018f0841600b95256187b9a08aeab2aa8b3c11]/fbcode/executorch/backends/xnnpack/test/test_xnnpack.py)
