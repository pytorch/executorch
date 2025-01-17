# Memory Planning

Audience: Backend integrators and embedded developers who are interested in customizing the regions of memory ExecuTorch programs operate in.

## Overview

MemoryPlanning is the very last action taken before taking an `ExportedProgram` and undergoing emission to an ExecuTorch program. During this process, ExecuTorch takes the size and lifespan of each mutable tensor, and plans out their location in fixed size memory arenas.

Concretely, there are three passes related to memory planning:
* `SpecPropPass` computes a TensorSpec for each tensor in the graph (inputs, intermediates or outputs). The most important field of the tensor spec is a symbolic expression of the shapes of the tensor, where the initial set of symbols comes from the dimensions of input tensors, intermediate tensor shapes’ symbolic expression is propagated via tensor operations. The dimensions can be marked as either dynamic or static by users and when the dims are dynamic, users are required to annotate the dim with a ValueRange.

* `SymShapeEvalPass` evaluates the symbolic expressions to concrete integers with their upper bounds. There are two ways to doing the upper bound specialization:
HintBasedSymShapeEval (to be deprecated) is the old way of evaluating the upper bound. It doesn’t look at the ValueRange of the symbols but uses the shapes of example inputs to replace all the symbols. We call it “hint based“ because the example inputs’ shapes are just hints of what the input shapes might be at run time and are used for tracing only. ValueRangeBasedSymShapeEval is the recommended way of doing UpperBoundMemory planning. It will actually look at the ValueRange of the symbols and do an inference over the ranges to get a real upper bound.

* `MemoryPlanningPass` does the actual memory planning given all tensors get a TensorSpec with concrete integer shapes.

## Algorithms

ExecuTorch provides two options for memory planning algorithms out of the box, but users can define their own if the provided options are inappropriate or insufficient for their use case.

* The naive algorithm simply concatenates all the tensors together in a linear memory block without considering memory re-use. It serves as an upper bound for total memory consumption and serves as a baseline.

* The Greedy algorithm tries to re-use the already allocated memory based on the best-fit criteria. Specifically:
When there isn’t an allocated memory whose lifetime doesn’t overlap with the current tensor that we try to do memory planning for, we allocate a new memory buffer with the same size and lifetime as the current tensor. When there is one or more allocated memory buffer, whose lifetime overlaps with the current tensor, we pick the buffer that has the closest size with current tensor so as to reduce memory fragmentation. Finally, we allocate these memory buffers linearly in memory.


## Method Inputs and Outputs

The `MemoryPlanningPass` exposes the option to not memory plan program inputs and outputs. If the IO is not planned then users will be expected to provide data buffers to back these values at runtime. Example:

```python
program = edge_program.to_executorch(
            exir.ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False, # Inputs will not be memory planned, the data_ptr for input tensors after model load will be nullptr
                    alloc_graph_output=True, # Outputs will be memory planned, the data_ptr for output tensors after model load will be in the `planned_memory`.
                )
            )
        )
```

One common set-up would be for models where the outputs of the model are provided as inputs to subsequent inferences. In that situation, it would generally be better to not memory plan the IO, and instead provide the same buffer to both the input and output at runtime to avoid a copy.

## Custom Memory Plans

Users can write custom memory plans to take advantage of multiple memory locations (like SRAM and DRAM), place the outputs of specific nodes in specific locations, or even change the planning algorithm itself. The following example shows how you could reuse the provided planning algorithms, but with multiple hierarchies and placing the outputs of specific ops in specific memory arenas.

```python
class CustomPoolMemoryPlanningPass(MemoryPlanningPass):
    def run(self, graph_module: GraphModule, graph_signature: Optional[ExportGraphSignature]) -> PassResult:
        for subgm in graph_module.modules():
            if not isinstance(subgm, GraphModule):
                continue
            for node in subgm.graph.nodes:
                # mem_id = 1 placeholder and outputs of mul
                # mem_id = 2 for outputs of add
                # parent class will copy spec will to alloc nodes
                if node.op == "placeholder":
                    node.meta["spec"].mem_id = 1
                    continue

                if node.op != "call_function":
                    continue

                if node.target == torch.ops.aten.add.out:
                    node.meta["spec"].mem_id = 2
                elif node.target == torch.ops.aten.mul.out:
                    node.meta["spec"].mem_id = 1

        return super().run(graph_module, graph_signature)
```

Then later when lowering to ExecuTorch you can use your custom plan in the following way:

```python
program = edge_program.to_executorch(
            exir.ExecutorchBackendConfig(
                memory_planning_pass=CustomPoolMemoryPlanningPass(
                    memory_planning_algo=greedy,
                )
            )
        )
```

Users attempting to write a custom memory planning algorithm should start by looking at [the greedy algorithm's implementation](https://github.com/pytorch/executorch/blob/d62c41ca86435e5316e7ed292b6d68aff27a2fb7/exir/memory_planning.py#L459C1-L459C12).

## Debugging Tool

Please refer to [Memory Planning Inspection](./memory-planning-inspection.md) for a tool to inspect the result of memory planning.
