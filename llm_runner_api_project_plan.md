# Decoder-only LLM Runner API Implementation Plan

## Project Overview
This project aims to implement a standardized C++ and Python API for running decoder-only LLMs in ExecuTorch, following designs that resemble HuggingFace transformers interfaces. The goal is to make it easier for users to run exported LLM models while allowing customization of components like KV cache management.

## Project Timeline
- Expected Start Date: April 15, 2025
- Expected End Date: May 15, 2025
- Total Duration: 4 weeks

## Task Breakdown

### Phase 1: Core C++ API Implementation (Week 1)

- [ ] **Task 1.1: Define and implement the IIOManager interface**
  - [ ] Create `IIOManager.h` with the interface declaration
  - [ ] Document the interface methods
  - [ ] Add appropriate namespacing and include guards

- [ ] **Task 1.2: Implement DefaultIOManager**
  - [ ] Create `DefaultIOManager.h` and `.cpp` 
  - [ ] Refactor/adapt functionality from existing `StaticAttentionIOManager`
  - [ ] Add unit tests for the implementation

- [x] **Task 1.3: Create GenerationConfig class**
  - [x] Define struct with appropriate parameters
  - [x] Include documentation for each parameter
  - [x] Implement reasonable defaults

- [x] **Task 1.4: Implement DecoderLLMRunner base class**
  - [x] Create header and implementation files
  - [ ] Implement the `load()` static factory method
  - [ ] Implement `generate()` method
  - [x] Implement supporting methods (stop, is_loaded)

### Phase 2: Integration with Existing Code (Week 2)

- [ ] **Task 2.1: Integrate with IRunner interface**
  - [ ] Update DecoderLLMRunner to work with existing IRunner
  - [ ] Handle error cases and exceptions properly

- [ ] **Task 2.2: Create LlamaRunner implementation**
  - [ ] Refactor existing code from examples/models/llama/runner/runner.cpp
  - [ ] Adapt to new API requirements
  - [ ] Add appropriate error handling

- [ ] **Task 2.3: Add comprehensive unit tests for C++ API**
  - [ ] Write tests for IIOManager implementations
  - [ ] Write tests for DecoderLLMRunner
  - [ ] Test error cases and edge conditions

- [ ] **Task 2.4: Update build system**
  - [ ] Update CMakeLists.txt to include new files
  - [ ] Handle dependencies appropriately
  - [ ] Ensure compatibility with existing build process

### Phase 3: Python Bindings Implementation (Week 3)

- [ ] **Task 3.1: Set up pybind11 framework for Python bindings**
  - [ ] Create binding file structure
  - [ ] Set up module definition

- [ ] **Task 3.2: Bind GenerationConfig to Python**
  - [ ] Create Python-accessible wrapper
  - [ ] Ensure proper type conversions
  - [ ] Add Python docstrings

- [ ] **Task 3.3: Bind IIOManager interface to Python**
  - [ ] Create abstract base class in Python
  - [ ] Ensure proper inheritance works with Python implementations

- [ ] **Task 3.4: Bind DefaultIOManager to Python**
  - [ ] Create Python-accessible wrapper
  - [ ] Add appropriate docstrings

- [ ] **Task 3.5: Bind DecoderLLMRunner to Python**
  - [ ] Expose load() static method
  - [ ] Expose generate() method with callback support
  - [ ] Add proper error handling for Python exceptions

- [ ] **Task 3.6: Create Python examples and tests**
  - [ ] Write example usage scripts
  - [ ] Create unit tests for Python API
  - [ ] Test custom Python KV cache manager implementations

### Phase 4: Documentation, Examples, and Final Integration (Week 4)

- [ ] **Task 4.1: Write comprehensive API documentation**
  - [ ] Document C++ API
  - [ ] Document Python API
  - [ ] Create API reference guide

- [ ] **Task 4.2: Create example applications**
  - [ ] Simple text generation example
  - [ ] Custom KV cache manager example
  - [ ] Integration with existing applications

- [ ] **Task 4.3: Performance optimization and benchmarking**
  - [ ] Identify and address performance bottlenecks
  - [ ] Compare with existing implementations
  - [ ] Document performance characteristics

- [ ] **Task 4.4: Final integration and testing**
  - [ ] Integration with existing executorch workflows
  - [ ] End-to-end testing
  - [ ] Address any final issues or bugs

- [ ] **Task 4.5: Update repository documentation**
  - [ ] Update README.md with new API information
  - [ ] Create tutorials for using the new API
  - [ ] Document migration path from old approach

## Deliverables

1. C++ implementation of the LLM Runner API
2. Python bindings for the API
3. Unit tests for both C++ and Python components
4. Example applications and usage guides
5. API documentation

## Dependencies

- ExecuTorch core libraries
- pybind11 for Python bindings
- Existing examples/models/llama implementation for reference
- pytorch-labs/tokenizers for tokenization functionality

## Risks and Mitigation

| Risk | Severity | Mitigation |
|------|----------|------------|
| Performance overhead from Python bindings | Medium | Profile and optimize critical paths, consider optional direct C++ path |
| Compatibility issues with existing code | High | Thorough testing with existing examples, maintain backward compatibility |
| Custom Python KV cache manager performance | Medium | Provide guidelines for performance-critical implementations, offer C++ fallbacks |
| API design limitations | Medium | Gather early feedback, be prepared to revise API before finalization |

## Future Considerations (Post-Implementation)

- Support for encoder-decoder architectures
- Support for multimodal models
- Integration with training and fine-tuning workflows
- Hardware-specific optimizations