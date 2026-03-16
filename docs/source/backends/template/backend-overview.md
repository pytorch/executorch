# Backend Template

Provide a brief overview/description of the backend. At a high-level, what does it do? Consider linking to top-level vendor documentation for the target hardware family and/or framework (Core ML, XNNPACK, etc.).

## Features

List high-level features of backend, such as operator and hardware support.

## Target Requirements

What hardware and software is required to run the backend on a specific device? For example, does it require specific iOS or Android OS versions? If it's an NPU, what hardware models are supported?

## Development Requirements

What software and hardware is needed to create a .PTE file targeting this backend? Are there any additional dependencies that need to be installed that are not included with the ExecuTorch pip package? How does the user install them?

## Using *Backend Name*

This section describes the steps users need to take in order to generate a .PTE targeting this backend. Include a full code sample for exporting and lowering a model to this backend. Make sure relevant imports for the backend partitioner are included.

## Runtime Integration

This section is intended to tell the user all of the steps they'll need to take to be able to run a .PTE file on-device that is targeting the given backend.
- What CMake targets should they link to?
- How is this backend compiled from source?
- Is the backend bundled by default in iOS and/or Android pre-built libraries?

## Reference

**→{doc}`backend-partitioner` — Partitioner options.**

**→{doc}`backend-quantization` — Supported quantization schemes.**

**→{doc}`backend-troubleshooting` — Debug common issues.**

**→{doc}`backend-arch-internals` — Backend internals.**

**→{doc}`tutorials/backend-tutorials` — Tutorials.**

**→{doc}`guides/backend-guides` — Tutorials.**

```{toctree}
:maxdepth: 2
:hidden:
:caption: {BACKEND} Backend

backend-troubleshooting
backend-partitioner
backend-quantization
backend-op-support
backend-arch-internals
tutorials/backend-tutorials
guides/backend-guides
```
