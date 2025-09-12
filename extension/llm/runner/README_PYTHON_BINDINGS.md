# Python Bindings for MultimodalRunner

## Overview

This project provides Python bindings for the ExecuTorch MultimodalRunner, enabling Python developers to easily use the multimodal LLM runner for processing mixed inputs (text, images, audio) and generating text outputs.

## Architecture

The MultimodalRunner is designed for Large Language Models that can process multimodal inputs and generate text outputs. It supports models like:
- LLaVA (vision-language models)
- CLIP-based models
- Speech-to-text models
- Other multimodal transformers

### Key Components

1. **MultimodalRunner** - Main runner class for multimodal inference
2. **MultimodalInput** - Handles different input modalities (text, image, audio)
3. **GenerationConfig** - Configuration for text generation parameters
4. **Stats** - Performance monitoring and statistics
5. **Tokenizer** - Text tokenization and decoding

## Project Structure

```
extension/llm/runner/
├── multimodal_runner_pybindings.cpp  # Python bindings implementation (NEW)
├── __init__.py                       # Python package initialization (NEW)
├── multimodal_runner.py              # Python wrapper classes (NEW)
├── utils.py                          # Utility functions (NEW)
├── CMakeLists.txt                    # Existing - update to include Python bindings
└── test/
    ├── test_multimodal_runner.py    # Unit tests for Python bindings (NEW)
    └── test_generation.py            # Generation tests (NEW)
    └── [existing test files]         # Existing C++ tests remain here
```

Note: We'll reuse the root-level `setup.py` and update the existing `CMakeLists.txt` rather than creating new ones.

## Action Items

### 1. Core Implementation Tasks

#### High Priority
- [x] ~~**Create Python bindings file** (`multimodal_runner_pybindings.cpp`)~~
  - [x] ~~Bind MultimodalRunner class~~
  - [x] ~~Bind MultimodalInput and helper functions~~
  - [x] ~~Bind GenerationConfig struct~~
  - [x] ~~Bind Stats class for performance monitoring~~
  - [x] ~~Implement error handling and exception translation~~

#### Medium Priority
- [x] ~~**Update existing CMakeLists.txt** in `extension/llm/runner/`~~
  - [x] ~~Add Python bindings target when EXECUTORCH_BUILD_PYBIND is enabled~~
  - [x] ~~Configure pybind11 integration~~
  - [x] ~~Link with extension_llm_runner library~~
  - [x] ~~Handle tokenizers dependency~~
  - [x] ~~Set up proper include paths~~

- [x] ~~**Update root-level setup.py**~~
  - [x] ~~Add multimodal_runner to the extensions list~~
  - [x] ~~Ensure proper build configuration~~
  - [x] ~~Handle platform-specific configurations~~

#### Low Priority
- [x] ~~**Create Python wrapper files** in `extension/llm/runner/`~~
  - [x] ~~`__init__.py` - Package initialization~~
  - [x] ~~`multimodal_runner.py` - High-level Python API~~
  - [x] ~~`utils.py` - Utility functions for input preprocessing~~

### 2. Build System Integration

- [ ] **Integrate with main CMake build**
  - [ ] Add Python bindings compilation when EXECUTORCH_BUILD_PYBIND is enabled
  - [ ] Update extension/llm/runner/CMakeLists.txt to build multimodal_runner_pybindings.cpp
  - [ ] Ensure proper dependency resolution

- [ ] **Handle dependencies**
  - [ ] Link against existing tokenizers Python bindings
  - [ ] Ensure Module and other dependencies are available
  - [ ] Handle pybind11 version requirements

### 3. Input/Output Handling

- [ ] **Implement MultimodalInput Python bindings**
  - [ ] Support for text inputs
  - [ ] Support for image inputs (numpy arrays, PIL Images)
  - [ ] Support for audio inputs (if applicable)
  - [ ] Mixed input ordering support

- [ ] **Implement callbacks**
  - [ ] Token generation callback
  - [ ] Statistics callback
  - [ ] Progress reporting

### 4. Testing and Documentation

- [ ] **Create comprehensive tests**
  - [ ] Unit tests for bindings
  - [ ] Integration tests with sample models
  - [ ] Performance benchmarks
  - [ ] Memory leak tests

- [ ] **Write documentation**
  - [ ] API documentation with examples
  - [ ] Installation guide
  - [ ] Usage tutorials
  - [ ] Model compatibility guide

### 5. Example Scripts

- [ ] **Create example scripts**
  - [ ] Basic text generation
  - [ ] Image + text (vision-language) example
  - [ ] Batch processing example
  - [ ] Streaming generation example

## Installation Instructions

### Prerequisites

- Python >= 3.8
- CMake >= 3.18
- C++17 compatible compiler
- PyTorch (for tensor operations)
- pybind11 >= 2.6.0

### Building from Source

```bash
# Clone the repository
git clone https://github.com/pytorch/executorch.git
cd executorch

# Install dependencies
pip install -r requirements.txt

# Build with Python bindings enabled
python setup.py install --cmake-args="-DEXECUTORCH_BUILD_PYBIND=ON"

# Or for development
pip install -e . --config-settings editable_mode=compat
```

### Running Tests

```bash
# Run the multimodal runner Python tests
python -m pytest extension/llm/runner/test/test_multimodal_runner.py -v
```

## Usage Example

```python
from executorch.extension.llm.runner import MultimodalRunner, GenerationConfig
from executorch.extension.llm.runner.utils import make_text_input, make_image_input
import numpy as np

# Initialize the runner
runner = MultimodalRunner(
    model_path="path/to/model.pte",
    tokenizer_path="path/to/tokenizer.bin"
)

# Create multimodal inputs
image_array = np.random.rand(224, 224, 3)  # Example image
inputs = [
    make_text_input("Describe this image:"),
    make_image_input(image_array)  # numpy array or PIL Image
]

# Configure generation
config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9
)

# Generate text with callbacks
def on_token(token):
    print(token, end='', flush=True)

def on_stats(stats):
    print(f"\nTokens/sec: {stats.tokens_per_second:.2f}")

runner.generate(inputs, config, token_callback=on_token, stats_callback=on_stats)

# Or simpler usage without callbacks
response = runner.generate_text(inputs, config)
print(response)
```

## Technical Considerations

### Memory Management
- Python bindings should properly handle memory ownership
- Use shared_ptr/unique_ptr appropriately
- Implement proper cleanup in destructors

### Threading and GIL
- Consider GIL release during long-running operations
- Ensure thread safety for callbacks
- Handle Python exceptions in C++ code

### Performance
- Minimize data copying between Python and C++
- Use move semantics where possible
- Consider zero-copy tensor operations

## Dependencies

### Required
- executorch core libraries
- extension_llm_runner
- tokenizers library
- pybind11

### Optional
- numpy (for array handling)
- PIL/Pillow (for image processing)
- torch (for tensor operations)

## Contributing

Please follow the ExecuTorch contribution guidelines. Key points:
- Code should be formatted with clang-format
- Python code should follow PEP 8
- Add comprehensive tests for new features
- Update documentation as needed

## License

This project is licensed under the BSD-style license found in the LICENSE file in the root directory of the ExecuTorch repository.

## Next Steps

1. **Review and approve this plan** with the team
2. **Start with core bindings** implementation
3. **Test with existing models** (LLaVA, etc.)
4. **Gather feedback** from early users
5. **Iterate and improve** based on usage patterns

## Questions for Discussion

1. Should we support async generation?
2. What level of integration with PyTorch tensors is needed?
3. Should we provide pre-built wheels or source-only distribution?
4. How should we handle model loading and caching?
5. What additional utilities would be helpful for users?