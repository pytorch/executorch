# Definition of the Core ATen Operator Set

This page provides the description and background of the Core ATen Operator Set (opset). This page is recommended reading for those developing a new kernel library or delegate for ExecuTorch. It is also recommended that one is familiar with [`torch.export`](https://pytorch.org/docs/main/export.html) as a prerequisite; in particular, the concepts of torch FX graphs, operator decomposition, and functionalization.

The list of operators that have been identified as a Core ATen operator can be found on the [IRs page of the PyTorch documentation website](https://pytorch.org/docs/main/torch.compiler_ir.html).

## What is an Operator Set?

`torch.export` performs a full graph capture on a given PyTorch program, producing a graph IR that describes the computation performed by the program. An operator (i.e. an operation performed on a Tensor) is the basic unit of computation in the graph, often corresponding to a unique node in the graph IR. The primary source of operators is the [ATen library](https://pytorch.org/cppdocs/#aten); outside of ATen operators, developers can also define their own operators (i.e. custom operators).

An “ATen operator set” or “ATen opset” is the set of ATen operators that can be used to represent a PyTorch program once it has been captured into a graph IR.

## The Functional ATen Operator Set

The program capture mechanism of `torch.export` produces a functionalized graph, which only allows functional operators (i.e. operators that do not mutate or alias inputs). Therefore, `torch.export` produces a graph that will contain the functional ATen opset, which contains only functional ATen operators.

## The Core ATen Operator Set

An exported graph can be further transformed by applying operator decompositions. This process will replace specified ATen operators with equivalent sequences of other ATen operators. For instance, `aten.hardsigmoid` can be replaced with `aten.clamp(aten.clamp(self + 3, min=0), max=6) / 6`.

If a PyTorch program is decomposed with the default decomposition settings, then the resulting graph IR will contain the “core ATen” opset. This opset will be a subset of the functional ATen opset, as some operators will be decomposed. ATen operators that are a part of the core ATen opset (i.e. core ATen operators) will not be decomposed under the default decomposition setting. Generally, core ATen operators cannot be easily re-expressed by other ATen operators through decomposition.

The key motivation behind the core ATen opset is to reduce the number of operators that need to be handled by PyTorch backends and compilers once a model is exported. Not only are there a great number of operators defined in the ATen library, but new operators may be added, or the schema of existing operators may change. Without operator decomposition, backends built on top of the IR produced by `torch.export` would have to deal with both a large operator surface, as well as an opset that is constantly in flux. The core ATen opset addresses this by defining a much smaller, more manageable set of operators that was developed with stability in mind.

## Development of the Core ATen Operator Set

Although ExecuTorch uses the core ATen opset, it is not specific to ExecuTorch. One of the primary design goals of the core ATen opset is that it should be as generic as possible; the vast majority of use-cases will not want to decompose the operators contained within it. By extension, the decompositions implied by the core ATen opset should be useful to the vast majority of use-cases.

Another key consideration was to keep the opset as minimal as possible, but not at the expense of imposing decompositions that would have a profound negative impact on performance or developer experience.

The core ATen opset was developed by reviewing a list of ATen operators created by surveying models in public GitHub repositories in addition to well-known open source models. The purpose of the surveying process was to obtain a reduced list of ATen operators that is a proxy of which ATen operators are used the most. This way the most commonly used operators may be reviewed first.

The decision of whether each operator should be a core operator or be decomposed by the Core ATen Decomposition Table was determined by:

1. Examining potential decompositions of the operators; the decomposition should be a relatively straightforward re-expression of the operator using other ATen operators.
    * The decomposition shouldn’t look like an outright implementation of the operator.
    * The decomposition shouldn't vary based on run-time characteristics of the input.
    * We also consider if decomposing the operator will impact the precision, numerical validity or memory layout of the output.
2. Thinking about whether developers would want to preserve the operator in the graph for performance or other reasons.
    * For instance, perhaps an operator can be decomposed but it can map to a single hardware instruction on most platforms, in which case it would be preferable to promote it to a core operator.

## Future Work

Until every ATen operator has been reviewed and given a designation of “core” or “decomposed by default”, the core ATen opset cannot be considered fully complete. However, this is a monumental task, and there is a long tail of operators that are not often used. This is why an approach was taken where models were surveyed to determine which ops were the most commonly used which allowed “higher impact” operators to be prioritized.

Nonetheless, there are still many operators which have not been evaluated. The plan is to continue evaluating additional operators as the need arises; the PyTorch community may propose additional core operators or additional core decompositions through opening a GitHub issue or by [commenting on this post on the PyTorch Forums](https://dev-discuss.pytorch.org/t/defining-the-core-aten-opset/1464).
