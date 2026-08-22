# AOTI Common Library

This directory contains **common library components** for AOTI (Ahead-of-Time Inference) driven backends in ExecuTorch, **not a standalone backend**.

## Purpose

The code in this directory provides shared functionality and utilities that are used by actual AOTI-driven backends such as:

- **CUDA backend** - Uses AOTI for GPU acceleration
- Other AOTI-powered backends

## Components

- **`common_shims.cpp/h`** - Common shim functions that bridge ExecuTorch tensor operations with AOTI requirements
- **`aoti_model_container.cpp/h`** - Model container functionality for AOTI models
- **`utils.h`** - Utility functions and type definitions
- **`tests/`** - Unit tests for the common functionality

## Usage

This library is intended to be used as a dependency by actual AOTI backend implementations. It is not a backend that can be used directly for model execution.

For example backend implementations that use this common library, see:
- `executorch/backends/cuda/` - CUDA AOTI backend

## Building

The common library components are built as part of the AOTI backend build process. See the `TARGETS` file for build configurations.
