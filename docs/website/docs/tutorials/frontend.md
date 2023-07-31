# Tutorial for creating a toy runtime for a subset of EXIR

## Introduction

The goal of EXIR and pytorch export in general is to capture the
computational graph and have it exported outside of pytorch.
Once exported, that graph can be consumed to perform a variety of tasks,
the most common being to generate a bytecode for a different runtime.

## Toy runtime

For simplicity, our toy runtime will only support scalars and have the following bytecode:

1. binary ops `+ - * /`: meaning pop 2 operands off the stack, and push the result into stack
2. A number: meaning to push that number into the stack.

I.e. our runtime is exact the reverse polish calculator as provided by `dc` Unix command.

## Front end:

The front end will be pytorch program that operates on scalar tensors.
i.e. we want to be able to compile simple programs like:

```python
from torch import nn

class MyModule(nn.Module):

    def forward(self, x, y):
        a = 2 * x
        b = 3 * y + x
        c = a + (x + y)
        return c
```

### Step 1. Produce EXIR via `exir.capture`

```python
# NOTE import path TBD
from executorch import exir

my_module = MyModule()

print(my_module(torch.Tensor(3), torch.tensor(4)))

my_module_exir = exir.capture(my_module, (torch.Tensor(3), torch.Tensor(4)))

print(my_module_exir.module.print_tabular())
```

`my_module_exir` above is an instance of `ExportProgram` instance, which have
a `module` field of type `torch.fx.GraphModule`.

### (optional) Step 2. Passes

A *pass* is a transform on EXIR Graphs. It can be thought as a function that takes
a `torch.fx.GraphModule` as input, modify it, and returns the modified `torch.fx.GraphModule`.

Usually a pass is used to implement an optimization, but can also be used for other things like
computing statistics or infering extra metadata.

For sake of example, we will implement a pass that replace the operation `2 * x` with `x + x`, because say,
in our toy runtime the operator of sum is much faster than multiplication.

You can see example of passes [here](../tutorials/passes.md).

### Step 3. Emit code

To create equivalent reverse polish notation code from our graph, we will create a function that takes
a `torch.fx.GraphModule` and return a list of characters that represents the bytecode of our runtime.

```python
def emit_rpn(graphmodule: torch.fx.GraphModule):
    # find output node

    code = []
    def emit_code_recursive(node):
        nonlocal code
        if isinstance(node, torch.Tensor):  # constant
            code.append(str(node))
        if node.op == 'call_function':
            for a in node.args:
                # first emit for all my operands
                emit_code_recursive(a)
            if node.target == torch.ops.aten.add:
                code.append('+')
            elif node.target == torch.ops.aten.add:
                code.append('+')
            elif node.target == torch.ops.aten.add:
                code.append('+')
            elif node.target == torch.ops.aten.add:
                code.append('+')
            else:
                raise AssertError('Unknown OP: ', node.target)

    output_node = [n for n in graphmodule.graph.nodes if n.op == 'output']
    emit_code_recursive(output_node[0].args[0])
    return code
```

### Step 4: Run the generated code in our runtime

```python

def run_bytecode(bytecode, *args):
    program = []
    program.extend(map(str, args))
    program.extend(bytecode)
    program.append('p') # p for print in dc
    os.system(['echo {} | dc'.format(' '.join(program))])
```
