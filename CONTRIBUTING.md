Thank you for your interest in contributing to ExecuTorch! We want to make
it easy to contribute to this project.


## Dev Install

Set up your environment by following the instructions at
https://pytorch.org/executorch/main/getting-started-setup to clone
the repo and install the necessary requirements.

Refer to this [document](docs/source/using-executorch-building-from-source.md) to build ExecuTorch from source.

### Dev Setup for Android
For Android, please refer to the [Android documentation](docs/source/using-executorch-android.md).

### Dev Setup for Apple
For Apple, please refer to the [iOS documentation](docs/source/using-executorch-ios.md).
&nbsp;

## Codebase structure

<pre>

executorch
├── <a href="backends">backends</a> - Backend delegate implementations for various hardware targets. Each backend uses partitioner to split the graph into subgraphs that can be executed on specific hardware, quantizer to optimize model precision, and runtime components to execute the graph on target hardware. For details refer to the <a href="docs/source/backend-delegates-integration.md">backend documentation</a> and the <a href="docs/source/using-executorch-export.md">Export and Lowering tutorial</a> for more information.
│   ├── <a href="backends/apple">apple</a> - Apple-specific backends.
│   │   ├── <a href="backends/apple/coreml">coreml</a> - CoreML backend for Apple devices. See <a href="docs/source/backends/coreml/coreml-overview.md">doc</a>.
│   │   └── <a href="backends/apple/mps">mps</a> - Metal Performance Shaders backend for Apple devices. See <a href="docs/source/backends/mps/mps-overview.md">doc</a>.
│   ├── <a href="backends/arm">arm</a> - ARM architecture backends. See <a href="docs/source/backends/arm-ethos-u/arm-ethos-u-overview.md">doc</a>.
│   ├── <a href="backends/cadence">cadence</a> - Cadence-specific backends. See <a href="docs/source/backends-cadence.md">doc</a>.
│   ├── <a href="backends/example">example</a> - Example backend implementations.
│   ├── <a href="backends/mediatek">mediatek</a> - MediaTek-specific backends. See <a href="docs/source/backends-mediatek.md">doc</a>.
│   ├── <a href="backends/openvino">openvino</a> - OpenVINO backend for Intel hardware.
│   ├── <a href="backends/qualcomm">qualcomm</a> - Qualcomm-specific backends. See <a href="docs/source/backends-qualcomm.md">doc</a>.
│   ├── <a href="backends/transforms">transforms</a> - Transformations for backend optimization.
│   ├── <a href="backends/vulkan">vulkan</a> - Vulkan backend for cross-platform GPU support. See <a href="docs/source/backends/vulkan/vulkan-overview.md">doc</a>.
│   └── <a href="backends/xnnpack">xnnpack</a> - XNNPACK backend for optimized neural network operations. See <a href="docs/source/backends/xnnpack/xnnpack-overview.md">doc</a>.
├── <a href="codegen">codegen</a> - Tooling to autogenerate bindings between kernels and the runtime.
├── <a href="configurations">configurations</a> - Configuration files.
├── <a href="devtools">devtools</a> - Model profiling, debugging, and inspection. Please refer to the <a href="docs/source/devtools-overview.md">tools documentation</a> for more information.
│   ├── <a href="devtools/bundled_program">bundled_program</a> - a tool for validating ExecuTorch model. See <a href="docs/source/bundled-io.md">doc</a>.
│   ├── <a href="devtools/etdump">etdump</a> - ETDump - a format for saving profiling and debugging data from runtime. See <a href="docs/source/etdump.md">doc</a>.
│   ├── <a href="devtools/etrecord">etrecord</a> - ETRecord - AOT debug artifact for ExecuTorch. See <a href="https://pytorch.org/executorch/main/etrecord">doc</a>.
│   ├── <a href="devtools/inspector">inspector</a> - Python API to inspect ETDump and ETRecord. See <a href="https://pytorch.org/executorch/main/model-inspector">doc</a>.
│   └── <a href="devtools/visualization">visualization</a> - Visualization tools for representing model structure and performance metrics.
├── <a href="docs">docs</a> - Static docs tooling and documentation source files.
├── <a href="examples">examples</a> - Examples of various user flows, such as model export, delegates, and runtime execution.
├── <a href="exir">exir</a> - Ahead-of-time library: model capture and lowering APIs. EXport Intermediate Representation (EXIR) is a format for representing the result of <a href="https://pytorch.org/docs/stable/export.html">torch.export</a>. This directory contains utilities and passes for lowering the EXIR graphs into different <a href="docs/source/ir-exir.md">dialects</a> and eventually suitable to run on target hardware.
│   ├── <a href="exir/_serialize">_serialize</a> - Serialize final export artifact.
│   ├── <a href="exir/backend">backend</a> - Backend delegate ahead of time APIs.
│   ├── <a href="exir/capture">capture</a> - Program capture.
│   ├── <a href="exir/dialects">dialects</a> - Op sets for various dialects in the export process. Please refer to the <a href="docs/source/ir-exir.md">EXIR spec</a> and the <a href="docs/source/compiler-backend-dialect.md">backend dialect</a> doc for more details.
│   ├── <a href="exir/emit">emit</a> - Conversion from ExportedProgram to ExecuTorch execution instructions.
│   ├── <a href="exir/operator">operator</a> - Operator node manipulation utilities.
│   ├── <a href="exir/passes">passes</a> - Built-in compiler passes.
│   ├── <a href="exir/program">program</a> - Export artifacts.
│   ├── <a href="exir/serde">serde</a> - Graph module serialization/deserialization.
│   ├── <a href="exir/verification">verification</a> - IR verification.
├── <a href="extension">extension</a> - Extensions built on top of the runtime.
│   ├── <a href="extension/android">android</a> - ExecuTorch wrappers for Android apps. Please refer to the <a href="docs/source/using-executorch-android.md">Android documentation</a> and <a href="https://pytorch.org/executorch/main/javadoc">Javadoc</a> for more information.
│   ├── <a href="extension/apple">apple</a> - ExecuTorch wrappers for iOS apps. Please refer to the <a href="docs/source/using-executorch-ios.md">iOS documentation</a> on how to integrate into Apple platform</a> for more information.
│   ├── <a href="extension/aten_util">aten_util</a> - Converts to and from PyTorch ATen types.
│   ├── <a href="extension/data_loader">data_loader</a> - 1st party data loader implementations.
│   ├── <a href="extension/evalue_util">evalue_util</a> - Helpers for working with EValue objects.
│   ├── <a href="extension/gguf_util">gguf_util</a> - Tools to convert from the GGUF format.
│   ├── <a href="extension/kernel_util">kernel_util</a> - Helpers for registering kernels.
│   ├── <a href="extension/llm">llm</a> - Library to run LLM on ExecuTorch including common optimization passes, runtime C++ components. Please refer to the <a href="docs/source/llm/getting-started.md">LLM documentation</a> for more information.
│   ├── <a href="extension/memory_allocator">memory_allocator</a> - 1st party memory allocator implementations.
│   ├── <a href="extension/module">module</a> - A simplified C++ wrapper for the runtime. An abstraction that deserializes and executes an ExecuTorch artifact (.pte file). Refer to the <a href="docs/source/extension-module.md">module documentation</a> for more information.
│   ├── <a href="extension/parallel">parallel</a> - C++ threadpool integration.
│   ├── <a href="extension/pybindings">pybindings</a> - Python API for executorch runtime. This is powering up the <a href="docs/source/runtime-python-api-reference.rst">runtime Python API</a> for ExecuTorch.
│   ├── <a href="extension/pytree">pytree</a> - C++ and Python flattening and unflattening lib for pytrees.
│   ├── <a href="extension/runner_util">runner_util</a> - Helpers for writing C++ PTE-execution tools.
│   ├── <a href="extension/tensor">tensor</a> - Tensor maker and <code>TensorPtr</code>, details in <a href="docs/source/extension-tensor.md">this documentation</a>. For how to use <code>TensorPtr</code> and <code>Module</code>, please refer to the <a href="docs/source/using-executorch-cpp.md">"Using ExecuTorch with C++"</a> doc.
│   ├── <a href="extension/testing_util">testing_util</a> - Helpers for writing C++ tests.
│   ├── <a href="extension/threadpool">threadpool</a> - Threadpool.
│   └── <a href="extension/training">training</a> - Experimental libraries for on-device training.
├── <a href="kernels">kernels</a> - 1st party kernel implementations.
│   ├── <a href="kernels/aten">aten</a> - ATen kernel implementations.
│   ├── <a href="kernels/optimized">optimized</a> - Optimized kernel implementations.
│   ├── <a href="kernels/portable">portable</a> - Reference implementations of ATen operators.
│   ├── <a href="kernels/prim_ops">prim_ops</a> - Special ops used in executorch runtime for control flow and symbolic primitives.
│   └── <a href="kernels/quantized">quantized</a> - Quantized kernel implementations.
├── <a href="profiler">profiler</a> - Utilities for profiling runtime execution.
├── <a href="runtime">runtime</a> - Core C++ runtime. These components are used to execute the ExecuTorch program. Please refer to the <a href="docs/source/runtime-overview.md">runtime documentation</a> for more information.
│   ├── <a href="runtime/backend">backend</a> - Backend delegate runtime APIs.
│   ├── <a href="runtime/core">core</a> - Core structures used across all levels of the runtime. Basic components such as <code>Tensor</code>, <code>EValue</code>, <code>Error</code> and <code>Result</code> etc.
│   ├── <a href="runtime/executor">executor</a> - Model loading, initialization, and execution. Runtime components that execute the ExecuTorch program, such as <code>Program</code>, <code>Method</code>. Refer to the <a href="https://pytorch.org/executorch/main/executorch-runtime-api-reference">runtime API documentation</a> for more information.
│   ├── <a href="runtime/kernel">kernel</a> - Kernel registration and management.
│   └── <a href="runtime/platform">platform</a> - Layer between architecture specific code and portable C++.
├── <a href="schema">schema</a> - ExecuTorch PTE file format flatbuffer schemas.
├── <a href="scripts">scripts</a> - Utility scripts for building libs, size management, dependency management, etc.
├── <a href="shim_et">shim_et</a> - Compatibility layer between OSS and Internal builds.
├── <a href="test">test</a> - Broad scoped end-to-end tests.
├── <a href="third-party">third-party</a> - Third-party dependencies.
├── <a href="tools">tools</a> - Tools for building ExecuTorch from source, for different built tools (CMake, Buck).
└── <a href="util">util</a> - Various helpers and scripts.
</pre>

&nbsp;

## Contributing workflow
We actively welcome your pull requests (PRs).

If you're completely new to open-source projects, GitHub, or ExecuTorch, please see our [New Contributor Guide](docs/source/new-contributor-guide.md) for a step-by-step walkthrough on making your first contribution. Otherwise, read on.

1. [Claim an issue](#claiming-issues), if present, before starting work. If an
   issue doesn't cover the work you plan to do, consider creating one to provide
   context about it, and to build consensus about the scope and solution.
1. Create your new branch from `main` in your forked repo, with a name
   describing the work you're completing; e.g., `add-feature-x`.
1. If you've added code that should be tested, add tests. Ensure all tests pass.
   See the [testing section](#testing) for more information.
1. If you've changed APIs or added a new tool or feature, [update the
   documentation](#updating-documentation).
1. If you added an experimental API or deprecated an existing API, follow the
   [API Life Cycle and Deprecation Policy](docs/source/api-life-cycle.md).
1. Make sure your code follows the [style guides](#coding-style) and passes the
   [lint checks](#lintrunner).
1. If you haven't already, complete the [Contributor License Agreement ("CLA")](#contributor-license-agreement-cla).
1. Create a pull request in the `pytorch/executorch` Github repo using the
   [instructions below](#pull-requests).

&nbsp;

## Issues

### Creating Issues
We use GitHub issues to track public bugs and feature requests. Ensure that the
issue title is clear and descriptive, and that the description has sufficient
instructions to be able to reproduce the issue.

Meta has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

### Issue Labels

#### Module/Partner Labels

[Labels beginning with `module:`](https://github.com/pytorch/executorch/labels?q=%22module%3A+%22)
indicate the area that the issue relates to. The ExecuTorch oncall will
typically add this label.

[Labels beginning with `partner:`](https://github.com/pytorch/executorch/labels?q=%22partner%3A+%22)
indicate the ExecuTorch partner who owns the issue. The ExecuTorch oncall will
typically add this label.

#### Lifecycle Labels

The ExecuTorch oncall will triage new issues. If the issue requires more
information from the issue's author, oncall will add the `need-user-input` label
and wait for the author to respond.

Once the issue contains enough information, the oncall will:
- Ensure that the title is descriptive
- Add one of the labels:
  - `bug`: The issue describes an unexpected problem
  - `feature`: The issue describes a request for new functionality
  - `rfc`: The issue describes a proposed change to functionality
- Add one `module:` label or one `partner:` label, as described above
- Add the `triaged` label

After this point, the oncall has finished the triage process, and the
module owner or partner is responsible for resolving the issue. (See
https://github.com/pytorch/executorch/issues/7679 for the mapping of labels to
owners.)

### Claiming Issues
We'd love your help closing out [open
issues](https://github.com/pytorch/executorch/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen)
in the Github repo.

1. Find an issue with the
   [`actionable`](https://github.com/pytorch/executorch/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen+label%3Aactionable)
   or [`good first
   issue`](https://github.com/pytorch/executorch/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
   label that is not currently assigned to anyone.
   - If you'd like to work on an issue that is assigned but hasn't been updated
     in a while, discuss a hand-off with the current assignee in the issue
     comments.
   - If you'd like to work on an issue that isn't marked `actionable`, please
     comment on the issue to ask about its status and wait for a response.
1. Set yourself as the assignee of the issue.
1. If you decide not to finish the issue, update the issue with information to
   help the next person, then remove yourself from the assignee list.
1. When creating pull requests (PRs), mention the issue number like `#1234` in
   the PR description details (the first comment in the PR conversation thread).
1. When the final PR has merged and resolves the issue, close the issue with the
   button at the bottom of the issue's page.

&nbsp;

## Coding Style

### lintrunner

We use [`lintrunner`](https://pypi.org/project/lintrunner/) to help make sure the
code follows our standards. Set it up with:

```
./install_requirements.sh  # (automatically run by install_executorch.sh)
lintrunner init
```

Then run `lintrunner` from the root of the repo to see its suggestions, or run
`lintrunner -a` to automatically apply the suggestions.

### Python Style

ExecuTorch Python code follows the style used by the PyTorch core project.

### C++ Style

ExecuTorch code uses the [Google C++
Style](https://google.github.io/styleguide/cppguide.html), with modifications.

Rationale: Google style is close to the C++ style used by PyTorch core, although
PyTorch core does not explicitly document its C++ style. Google style is well
documented, and has exceptional tooling support.

**Modifications** to the Google C++ style, to make it closer to the code in
PyTorch core:
- Function and method names should use `lower_snake_case()`. This follows the
  convention that PyTorch core inherited from its namesake Python, and is the
  biggest modification to the Google C++ style.
- File names should use `lower_snake_case.cpp` (not `.cc`, and not
  `PascalCase.cpp`). This follows the most common pattern in PyTorch core.
- Headers should use `#pragma once` instead of manual include guards. This
  follows the most common pattern in PyTorch core.
- All includes should use `<angle brackets>`, not `"double quotes"`. This
  ensures that headers are included using the compiler's include path, and not
  relative to the local file.
- Documentation comments should follow Doxygen syntax, either `//** ... */`
  (multi-line) or `/// ...` (single line), with `@`-style parameters like
  `@param`, `@retval`. Public APIs must be documented in the `.h` files that
  declare them.
- TODOs should prefer to reference a task or issue number like `TODO(#123):
  <description>`, rather than a username. A task can manage much-more-nuanced
  information, and can change ownership as people leave and join the project.

See the rest of this file for other portability- and efficiency-related
modifications to the Google C++ style guide.

### C++ Portability Guidelines

See also [Portable C++ Programming](docs/source/portable-cpp-programming.md)
for detailed advice.

#### C++ language version

**C++17.**

Rationale: This is a compromise between being compatible with older, proprietary
toolchains, and having access to relatively modern C++ features.

#### C/C++ standard library usage

**Restricted usage of the C++ standard library**

Rationale: ExecuTorch is intended to be portable to bare-metal systems that lack
certain features, like dynamic memory, threading, and locking, required by parts
of the standard library. It is also intended to be as small as possible, and
some convenient stdlib features may grow the binary size unacceptably.

Generally, do not instantiate types that allocate memory under the hood, like
`std::vector` or `std::string`. Do not call `new`, `malloc()` or `mmap()`; do
not use iostreams; do not operate on files.

However, it is convenient and portable (and sometimes necessary) to use static
standard library concepts like `std::move`, or metaprogramming helpers like
`std::is_floating_point<>`.  Pure code like `<cmath>` and `<cstring>` is fine,
as long as you stay away from functions that allocate memory (like `strdup()`).

It is also allowed (and sometimes necessary) to use "placement `new`", but be
careful to also manually destroy objects initialized in this way.

#### C++ language features

**Exceptions: Do not use**
- Rationale: Exceptions are not widely supported on some classes of
  microcontrollers and DSPs, and they can significantly increase binary size.

**Threads, thread_local, locking: Do not use, except in optional libraries that
must work with threading**
- Rationale: The core runtime must work on systems that do not have threading
  support.

**RTTI, dynamic_cast, and `<typeid>`: Do not use**
- Rationale: RTTI adds extra data to every virtual class. ExecuTorch doesn't
  have a strong need for `dynamic_cast` and friends, so it's better to reduce
  the binary size.

**Templates and template metaprogramming: Be careful and avoid if possible**
- Rationale: Most templating results in code generation, and is one of the most
  common sources of binary bloat. Some use of templates is fine (e.g. an
  `ArrayRef<T>`, or code that handles multiple `ScalarType` types), but for the
  most part avoid them if possible.

&nbsp;

## Testing

### Running Tests Locally

CI is run automatically on all pull requests. However, if you want to run tests locally, here are some example commands (not exhaustive):

- The `sh test/build_size_test.sh` script will compile the C++runtime along with portable kernels.
- The `test/run_oss_cpp_tests.sh` script will build and run C++ tests locally
- Running `pytest` from the root directory will run Python tests locally. Make sure to run this after finishing [Dev Install](#dev-install).

### Writing Tests
To help keep code quality high, ExecuTorch uses a combination of unit tests and
end-to-end (e2e) tests. If you add a new feature or fix a bug, please add tests
to ensure that the feature/fix works properly and continues to work properly.

Most directories in the repo already contain test files. In many cases, you can
add a test to an existing file, and the existing CI jobs will run it will run
automatically. If you do this, please take a look at the CI job logs to ensure
that it did actually run.

If it's not clear how to add a test for your PR, take a look at the blame for
the code you're modifying and find an author who has more context. Ask them
for their help in the PR comments.

### Continuous Integration
See https://hud.pytorch.org/hud/pytorch/executorch/main for the current state of
the CI (continuous integration) jobs. If `main` is broken, consider rebasing
your PR onto the `release/1.1` branch, which points to the most recent
all-green commit.

&nbsp;

## Updating Documentation

### APIs
ExecuTorch documents its APIs using inline code comments: doc strings for
Python, and Doxygen comments for C++. When modifying or adding an API, be sure
to modify or add documentation to the interfaces that you change. If the API
doesn't have inline documentation yet, please help improve the code by adding
documentation and describing the rest of the piece you modified.

Also search for references to the API you modified under `docs/source` to see if
any docs need to be modified to reflect your changes; these are the files that
are published on https://pytorch.org/executorch. If you are adding a new API,
look for places in the docs that would benefit from talking about that API, or
even create a new document for it. A job on the PR will give you a link to a
website preview based on your changes.

&nbsp;

## Pull Requests
This repo uses Github pull requests (PRs) to stage and review code before
merging it into the `main` branch. See the [Github
docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)
for basics.

1. Push your branch to your fork of `pytorch/executorch`. Most people do not
  have permission to push a branch directory to the upstream repo.
1. Create your PR
   - Use the `main` branch as the base.
   - Give the PR a clear and descriptive title. It will become the title of the
     merged commit, so it needs to be useful in the output of `git log`.
     - Bad title: "Fix a bug"
     - Good title: "Add XYZ method to ABC"
   - Give the PR a clear and thorough description. Don't just describe what the PR
     does: the diff will do that. Explain *why* you are making this change, in a
     way that will make sense to someone years from now. If the PR is a bug fix,
     include the issue number at the beginning of the description: "Fixes #1234"
   - Explain how you have tested your changes by including repeatable instructions for
     testing the PR.
     - If you added tests, this can be as simple as the command you used to run the
       tests.
     - If you tested the PR manually, include the steps and the outputs. Help a
       future editor understand how to test the code that you're modifying
       today.
   - If your PR contains or is representative of a feature/bug fix that should be
     called out in the release notes, please add a label for "Release notes: \<area\>",
	 where \<area\> describes which part of ExecuTorch the change pertains to, e.g.
	 "Release notes: runtime". Here are all of the categories:
     - `Release notes: runtime`: changes related to the core runtime which loads the program methods, initializes delegates, and runs the lowered graph.
     - `Release notes: exir`: changes to any internal representations, such as any edge-related dialects. Also any changes to passes that may modify the exir, such as memory planning.
     - `Release notes: quantization`: changes to quantization.
     - `Release notes: ops & kernels`: changes to the opset and any new / changed kernel implementations.
     - `Release notes: api`: changes to public facing apis (any interfaces, pybinded runtime methods, etc.).
     - `Release notes: backends`: changes to any of the backend delegates.
     - `Release notes: build`: changes related to the build system, including major dependency upgrades, notable build flags, optimizations, etc.
     - `Release notes: devtools`: changes to any of ExecuTorch's developer tools, for example the debugger & profiler.
     - `Release notes: examples`: changes to any code under `examples/`.
     - `Release notes: misc`: anything notable that doesn't belong in the above categories.
   - See https://github.com/pytorch/executorch/pull/3612 for an example PR that
     follows this advice.
1. Before asking for a review, ensure that all [CI (continuous integration)
   jobs](#continuous-integration) on your pull request succeed.
   - If the jobs on your PR are broken but you're not sure why, add a comment
     and proceed to finding a reviewer.
   - Not all users can trigger the CI jobs. If the jobs don't run on your PR,
     proceed to finding a reviewer.
1. Find reviewers
   - If you have been working with a member of the ExecuTorch repo, add them
     as a reviewer (*not* an "assignee").
   - If not, look at the blame for the files that the PR modifies, and try
     picking one or two ExecuTorch repo members as reviewers (*not*
     "assignees").
   - If you are unsure, leave a comment on the PR and keep it unassigned with no
     reviewers. A member of the ExecuTorch repo will find someone to review it.
1. Address and discuss comments left by reviewers
   - If the reviewers have requests or questions, follow up with them.
   - The goal of the reviewer is to ensure that the code in the `main` branch of
     the repo is consistent, maintainable, and of high quality.
1. Once the PR has been approved, you can merge it yourself
     by clicking the "Squash and merge" button once it is
     green and all CI signals are passing.

&nbsp;

## For Backend Delegate Authors

- Use [this](docs/source/backend-delegates-integration.md) guide when
  integrating your delegate with ExecuTorch.
- Refer to [this](docs/source/backend-delegates-dependencies.md) set of
  guidelines when including a third-party dependency for your delegate.

&nbsp;

## License
By contributing to ExecuTorch, you agree that your contributions will be
licensed under the LICENSE file in the root directory of this source tree.

&nbsp;

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

&nbsp;
