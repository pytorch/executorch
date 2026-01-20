# ExecuTorch VCPKG Port

This directory contains the VCPKG port configuration for ExecuTorch, PyTorch's on-device AI inference framework.

## About VCPKG Ports

VCPKG is a C/C++ package manager from Microsoft. A "port" is a set of scripts and metadata that tells VCPKG how to download, build, and install a library.

## Files in this Directory

- **vcpkg.json**: Manifest file containing package metadata, version, dependencies, and optional features
- **portfile.cmake**: Main build script that tells VCPKG how to configure, build, and install ExecuTorch
- **usage**: Instructions for users on how to use the installed package in their CMake projects

## Using this Port

### For VCPKG Registry Maintainers

To submit this port to the official VCPKG registry:

1. Follow the VCPKG contribution guidelines: https://github.com/microsoft/vcpkg/blob/master/docs/maintainers/control-files.md
2. Copy the `ports/executorch` directory to your VCPKG installation's `ports/` directory
3. Update the `SHA512` hash in `portfile.cmake` after the first build attempt
4. Test the port locally: `vcpkg install executorch`
5. Submit a pull request to the VCPKG repository

### For Local Development

To use this port locally without submitting to the registry:

1. Copy the `ports/executorch` directory to your VCPKG installation's `ports/` directory
2. Install: `vcpkg install executorch`
3. Use in your project as shown in the `usage` file

## Features

The port supports several optional features:

- **xnnpack**: XNNPACK backend for accelerated inference
- **coreml**: CoreML backend (Apple platforms only)
- **mps**: Metal Performance Shaders backend (Apple platforms only)
- **vulkan**: Vulkan backend for GPU acceleration
- **qnn**: Qualcomm QNN backend
- **portable-ops**: Portable CPU operators
- **optimized-ops**: Optimized CPU operators
- **quantized-ops**: Quantized operators
- **pybind**: Python bindings
- **tests**: Build and run tests

To install with features:
```bash
vcpkg install executorch[portable-ops,xnnpack]
```

## Maintenance

When a new version of ExecuTorch is released:

1. Update the `version-string` in `vcpkg.json`
2. Update the `REF` in `portfile.cmake` to point to the new release tag
3. Run `vcpkg install executorch` and update the `SHA512` hash based on the error message
4. Test the build
5. Update the VCPKG registry

## Documentation

- ExecuTorch: https://pytorch.org/executorch/
- VCPKG Getting Started: https://learn.microsoft.com/en-us/vcpkg/get_started/get-started
- VCPKG Packaging Guide: https://learn.microsoft.com/en-us/vcpkg/get_started/get-started-packaging
