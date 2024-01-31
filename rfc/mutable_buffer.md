## Mutable Buffers In ExecuTorch [RFC]

Author: Jacob Szwejbka
Last Update 1/31/24



This doc is a proposed solution to adding support for models with mutable buffers in the ExecuTorch runtime.


## Context:

**Q**: What is a mutable buffer?

**A**: A mutable buffer is basically any state inside a PyTorch program that is mutated across inferences. It is the state of a stateful model. A common version of this in practice is the KV-cache of an LLM. Here is a toy example of a stateful model:

```
class Model(torch.nn.Module):
    def __init__(self):
       super().__init__()
       self.state = torch.zeros(1)

    def forward(self, x):
	y = x + self.state
	self.state.add_(1)
       return y
```

Running export on this generates this graph

```
graph():
    %arg0_1 : [num_users=2] = placeholder[target=arg0_1]
    %l_x_ : [num_users=1] = placeholder[target=l_x_]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%l_x_, %arg0_1), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, 1), kwargs = {})
    return (add_1, add)
```

Here self.state has been lifted out of the model graph and is passed in as input “arg0_1” and the updated version is returned as output “add_1”.

This lifting is generally hidden from the user and inference entry points such as ExportedProgram.forward manually do the write back step of “self.state_dict[“arg0_1”] = add_1” before the next inference.

Code pointer: [https://github.com/pytorch/pytorch/blob/main/torch/export/exported_program.py#L315](https://github.com/pytorch/pytorch/blob/main/torch/export/exported_program.py#L315)


## Problem #1 (The Write-back):

In ExecuTorch the runtime cannot safely perform the write-back step of “self.state_dict[“arg0_1”] = add_1”. ExecuTorch memory planning allows users to provide custom memory buffers to back their tensors and there is no guarantee that the runtime can safely dereference the pointers and access these buffers’ content.

Normally this is alright because the runtime never has to access these buffers, and users ahead of time know that certain tensors will be on certain memory locations and can provide custom operators or custom operator implementations to handle these problem tensors for normal graph operations like torch.add etc.

Here though the write back happens outside the graph meaning it's impossible for users to target the copy node and replace it with something they know how to handle.


## Proposal for #1:


##### Context:

The information of what outputs and inputs are connected as mutable buffers is still present in the ExportedProgram in the GraphSignature. For the above example we have a graph signature of:

```
ExportGraphSignature(
  input_specs=[
    InputSpec(kind=<InputKind.BUFFER: 3>, arg=TensorArgument(name='arg0_1'),         target='L__self___state'),
         InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='l_x_'), target=None)
  ],
   output_specs=[
     OutputSpec(kind=<OutputKind.BUFFER_MUTATION: 3>, arg=TensorArgument(name='add_1'), target='L__self___state'),
           OutputSpec(kind=<OutputKind.USER_OUTPUT: 1>, arg=TensorArgument(name='add'), target=None)
  ]
)
```

Here we can see that the first InputSpec and OutputSpec are Buffer and BufferMutation both with the corresponding target `'L__self___state'`


##### Solution:

Inside of “to_edge” during the transformation from ATen to Edge dialect we will re-inject these mutations into the graph at the very end. Here is a rough outline of the proposed graph (using the same model as above):

```
graph():
    %arg0_1 : [num_users=2] = placeholder[target=arg0_1]
    %l_x_ : [num_users=1] = placeholder[target=l_x_]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%l_x_, %arg0_1), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, 1), kwargs = {})
  %copy_1: [num_users=1] =    call_function[target=torch.ops.aten.copy_](args=%arg0_1, %add_1)
    return (copy_1, add)
```

Now that the copy nodes are explicit in the graph again users can target them ahead of time to be replaced with ops that know how to handle their custom memory heirarchies.


## Problem #2 (Model Load):

Torch.export captures the state of buffers at export time. This means users can have buffers with a non-trivial initial state (as opposed to just all zeros). These buffers contents will have to be stored in the ExecuTorch flatbuffer, and then at method load time will have to be copied out of the flatbuffer into the mutable memory provided through memory planning and the non_const buffers.

This results in a similar problem to #1 where again the ExecuTorch runtime needs to access the memory backing tensors. It's also harder to handle this copy ahead of time because the ExportedProgram doesn’t really have a representation for what needs to happen at model init time.

## Proposal for #2:

At allocator creation time there is an optional argument allowing users to provide a function to memcpy from flatbuffer memory into the specified memory_id.

```
typedef void* (*MemcpyFn)(void* src, void* dst, size_t len);
struct MemoryBank {
  Span<uint8_t> buffer;
  Optional<MemcpyFn> MemcpyFn;
  // Could add new per-bank options here in the future
}

// New ctor
HierarchicalAllocator(Span<MemoryBank>);
```


Providing this function is only necessary from the user if the runtime needs to behave in a special manner when dereferencing the memory. If nothing is provided the runtime will default to std::memcpy or something equivalent.


##### Other Optimizations:

We should only require storing buffer contents in the flatbuffer and copying at model load time for buffer with a meaningful initial state. All zero buffers where initial state doesn't matter should be a fairly common case ahead of time that we can use heuristically for now, and for them we can provide better space optimization by only storing the buffer metadata (shape, dtype) in the flatbuffer, and a better runtime experience by not requiring users to provide a mem_copy implementation for custom.


##### Internal .pte Changes (Partners can skip this section)

Add a new case to allocation information for mutable buffers with initial state. The index will point to different tables depending on if it's a mutable buffer or a constant weight. Mutable buffers without an initial state can be treated the same as all other activations or non_const at this stage.

```
// In summary:
//   1. data_buffer_idx > 0, pre_allocation = Null: Tensor is a constant
//   2. data_buffer_idx = 0, pre_allocation = Non Null: Tensor is a non-constant.
//   3. data_buffer_idx = 0, pre_allocation = Null: Tensor is a non-constant
//     that will receive a dataptr at input time or during execution.
//   4. data_buffer_idx > 0, pre_allocation = Non Null: Tensor is a mutable buffer with initial state
//
// Index to the program's constants and buffers tables, value 0 is reserved
  data_buffer_idx:uint;
```

Add a buffer table to store the initial states to table Program

```
mutable_buffer_data:[Buffer] # can also be in a segment
```


## What about Delegates?:



* If the delegate is the sole user of a buffer and can handle all the operations that take place on a buffer then they can just consume it entirely and the rest of this doc doesn't really matter. The buffer becomes mostly an implementation detail of the delegate.
* If the delegate is the sole user of a buffer but cannot perform all the needed operations to the buffer (such as index_put in the case of kv-cache) then it must leave the buffer as io to the call_delegate.
* If the delegate is one of many users of the buffer then there is nothing really special about the situation. The delegate must treat it as if it was one of the many users of an activation and ensure whatever it takes in and returns follows the contracts of the rest of the program.


## Additional Thoughts:

The above problems and proposals are to guarantee safety and correctness of the executing program. There is also the consideration of performance. In the case of the kv-cache for instance the state is huge, and the operations on it are relatively simple. In scenarios like that a functional graph has considerable cost as we can have multiple copies of the cache taking up space in memory and have to perform large copies. To avoid this we should explore allowing in-place ops to exist in the graph as well as views/aliasing if they are contiguous.

We can do this slowly since it should only affect performance not correctness and start from only reintroducing these if they won't have downstream side effects later passes need to be mindful of.
