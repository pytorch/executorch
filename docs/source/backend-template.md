# Backend Template

Provide a brief overview/description of the backend. At a high-level, what does it do? Consider linking to top-level vendor documentation for the target hardware family and/or framework (Core ML, XNNPACK, etc.).

## Features

List high-level features of backend, such as general operator and hardware support.

## Target Requirements

What hardware and software is required to run the backend on a specific device? For example, does it require specific iOS or Android OS versions? If it's an NPU, what hardware models are supported?

## Development Requirements

What software and hardware is needed to create a .PTE file targeting this backend? Are there any additional dependencies that need to be installed that are not included with the ExecuTorch pip package? How does the user install them?

## Using *Backend Name*

This section describes the steps users need to take in order to generate a .PTE targeting this backend. Include a full code sample for exporting and lowering a model to this backend. Make sure relevant imports for the backend partitioner are included.

### Partitioner API

What options, if any, does the partitioner take? Are there any other export-time configurations that can be applied? Document each option.

### Quantization

What quantization schemes does this backend support? Consider including the following, as appropriate.
- What operators are supported?
- Number of bits?
- Static vs dynamic activations?
- Weight only vs activations + weights?
- Symmetric vs asymmetric weights?
- Per-tensor, per-chanel, group/blockwise?

If using a PT2E quantizer, document how to initialize the quantizer and all relevant configs and options.

Include a code snippet demonstrating how to perform quantization for this backend. Document, or link to, a description of the parameters that the user can specify.

## Runtime Integration

This section is intended to tell the user all of the steps they'll need to take to be able to run a .PTE file on-device that is targeting the given backend.
- What CMake targets should they link to?
- How is this backend compiled from source?
- Is the backend bundled by default in iOS and/or Android pre-built libraries?
