# Operator Compliance Test Suite

This directory contains operator tests that all backends are expected to pass. While not every backend will implement every operator or permutation, the expectation is that backend partitioners will only partition nodes that the backend can support. The partitioner should never error out due to not supporting an input node.

## Backend Registration

To plug into the test framework, each backend should provide an implementation of the Tester class, defined in backends/test/harness/tester.py. Backends can provide implementations of each stage, or use the default implementation, as appropriate.

At a minimum, the backend will likely need to provide a custom implementation of the Partition and ToEdgeTransformAndLower stages using the appropriate backend partitioner. See backends/xnnpack/test/tester/tester.py for an example implementation.

Once a tester is available, the backend flow(s) can be added in __init__.py in this directory by adding an entry to `ALL_TESTER_FLOWS`. Each flow entry consists of a name (used in the test case naming) and a function to instantiate a tester for a given model and input tuple.

## Test Cases

Operator test cases are defined under the operators/ directory. Tests are written in a backend-independent manner, and each test is programmatically expanded to generate a variant for each registered backend flow. The `@operator_test` decorator is applied to each test class to trigger this behavior. Tests can also be tagged with an appropriate type specifier, such as `@dtype_test`, to generate variants for each dtype. The decorators and "magic" live in __init__.py in this directory.
