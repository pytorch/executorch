# ==============================================================================
# ExecuTorch Targets Makefile
# ==============================================================================
#
# This Makefile provides convenient targets for building ExecuTorch model runners
# with different backend configurations (CPU, CUDA, Metal), as well as other
# binary targets.
#
# WHAT THIS BUILDS:
# -----------------
# Each target builds:
# 1. ExecuTorch core libraries with the specified backend (CPU, CUDA, or Metal)
# 2. The model-specific runner executable in cmake-out/examples/models/<model>/
#
# SUPPORTED MODELS:
# -----------------
# - voxtral:  Multimodal voice + text model (CPU, CUDA, Metal)
# - whisper:  Speech recognition model (CPU, CUDA, Metal)
# - llama:    Text generation model (CPU)
# - llava:    Vision + language model (CPU)
# - gemma3:   Text generation model (CPU, CUDA)
#
# USAGE:
# ------
# make <model>-<backend>    # Build a specific model with a backend
# make help                  # Show all available targets
# make clean                 # Remove all build artifacts
#
# Examples:
#   make voxtral-cuda        # Build Voxtral with CUDA backend
#   make llama-cpu           # Build Llama with CPU backend
#   make whisper-metal       # Build Whisper with Metal backend (macOS)
#
# HOW TO ADD A NEW MODEL:
# -----------------------
# To add a new model (e.g., "mymodel"), follow these steps:
#
# 1. Create a CMakePresets.json in examples/models/mymodel/:
#    - Define configurePresets for each backend (base, cpu, cuda, metal)
#    - Define buildPresets with the target name from CMakeLists.txt
#    - Define workflowPresets that combine configure + build steps
#    - See examples/models/voxtral/CMakePresets.json for multi-backend reference
#    - Or see examples/models/llama/CMakePresets.json for simple single-preset reference
#
# 2. Add targets to this Makefile:
#    a) Add to .PHONY declaration: mymodel-cuda mymodel-cpu mymodel-metal
#    b) Add help text in the help target
#    c) Add target implementations following this pattern:
#
#       mymodel-cuda:
#           @echo "==> Building and installing ExecuTorch with CUDA..."
#           cmake --workflow --preset llm-release-cuda
#           @echo "==> Building MyModel runner with CUDA..."
#           cd examples/models/mymodel && cmake --workflow --preset mymodel-cuda
#           @echo ""
#           @echo "✓ Build complete!"
#           @echo "  Binary: cmake-out/examples/models/mymodel/mymodel_runner"
#
#       mymodel-cpu:
#           @echo "==> Building and installing ExecuTorch..."
#           cmake --workflow --preset llm-release
#           @echo "==> Building MyModel runner (CPU)..."
#           cd examples/models/mymodel && cmake --workflow --preset mymodel-cpu
#           @echo ""
#           @echo "✓ Build complete!"
#           @echo "  Binary: cmake-out/examples/models/mymodel/mymodel_runner"
#
#       mymodel-metal:
#           @echo "==> Building and installing ExecuTorch with Metal..."
#           cmake --workflow --preset llm-release-metal
#           @echo "==> Building MyModel runner with Metal..."
#           cd examples/models/mymodel && cmake --workflow --preset mymodel-metal
#           @echo ""
#           @echo "✓ Build complete!"
#           @echo "  Binary: cmake-out/examples/models/mymodel/mymodel_runner"
#
# 3. Test your new targets:
#    make mymodel-cpu     # or mymodel-cuda, mymodel-metal
#
# NOTES:
# ------
# - CUDA backend is only available on Linux systems
# - Metal backend is only available on macOS (Darwin) systems
# - Some models may not support all backends (check model documentation)
# - Binary outputs are located in cmake-out/examples/models/<model>/
# - The preset names in CMakePresets.json must match the names used in Makefile
#
# ==============================================================================

.PHONY: voxtral-cuda voxtral-cpu voxtral-metal whisper-cuda whisper-cpu whisper-metal llama-cpu llava-cpu gemma3-cuda gemma3-cpu executor-runner-xnnpack clean help

help:
	@echo "This Makefile adds targets to build runners for various models on various backends. Run using `make <target>`. Available targets:"
	@echo "  voxtral-cuda           - Build Voxtral runner with CUDA backend"
	@echo "  voxtral-cpu            - Build Voxtral runner with CPU backend"
	@echo "  voxtral-metal          - Build Voxtral runner with Metal backend (macOS only)"
	@echo "  whisper-cuda           - Build Whisper runner with CUDA backend"
	@echo "  whisper-cpu            - Build Whisper runner with CPU backend"
	@echo "  whisper-metal          - Build Whisper runner with Metal backend (macOS only)"
	@echo "  llama-cpu              - Build Llama runner with CPU backend"
	@echo "  llava-cpu              - Build Llava runner with CPU backend"
	@echo "  gemma3-cuda            - Build Gemma3 runner with CUDA backend"
	@echo "  gemma3-cpu             - Build Gemma3 runner with CPU backend"
	@echo "  executor-runner-xnnpack - Build generic executor_runner with XNNPack backend"
	@echo "  clean                  - Clean build artifacts"

voxtral-cuda:
	@echo "==> Building and installing ExecuTorch with CUDA..."
	cmake --workflow --preset llm-release-cuda
	@echo "==> Building Voxtral runner with CUDA..."
	cd examples/models/voxtral && cmake --workflow --preset voxtral-cuda
	@echo ""
	@echo "✓ Build complete!"
	@echo "  Binary: cmake-out/examples/models/voxtral/voxtral_runner"

voxtral-cpu:
	@echo "==> Building and installing ExecuTorch..."
	cmake --workflow --preset llm-release
	@echo "==> Building Voxtral runner (CPU)..."
	cd examples/models/voxtral && cmake --workflow --preset voxtral-cpu
	@echo ""
	@echo "✓ Build complete!"
	@echo "  Binary: cmake-out/examples/models/voxtral/voxtral_runner"

voxtral-metal:
	@echo "==> Building and installing ExecuTorch with Metal..."
	cmake --workflow --preset llm-release-metal
	@echo "==> Building Voxtral runner with Metal..."
	cd examples/models/voxtral && cmake --workflow --preset voxtral-metal
	@echo ""
	@echo "✓ Build complete!"
	@echo "  Binary: cmake-out/examples/models/voxtral/voxtral_runner"

whisper-cuda:
	@echo "==> Building and installing ExecuTorch with CUDA..."
	cmake --workflow --preset llm-release-cuda
	@echo "==> Building Whisper runner with CUDA..."
	cd examples/models/whisper && cmake --workflow --preset whisper-cuda
	@echo ""
	@echo "✓ Build complete!"
	@echo "  Binary: cmake-out/examples/models/whisper/whisper_runner"

whisper-cpu:
	@echo "==> Building and installing ExecuTorch..."
	cmake --workflow --preset llm-release
	@echo "==> Building Whisper runner (CPU)..."
	cd examples/models/whisper && cmake --workflow --preset whisper-cpu
	@echo ""
	@echo "✓ Build complete!"
	@echo "  Binary: cmake-out/examples/models/whisper/whisper_runner"

whisper-metal:
	@echo "==> Building and installing ExecuTorch with Metal..."
	cmake --workflow --preset llm-release-metal
	@echo "==> Building Whisper runner with Metal..."
	cd examples/models/whisper && cmake --workflow --preset whisper-metal
	@echo ""
	@echo "✓ Build complete!"
	@echo "  Binary: cmake-out/examples/models/whisper/whisper_runner"

llama-cpu:
	@echo "==> Building and installing ExecuTorch..."
	cmake --workflow --preset llm-release
	@echo "==> Building Llama runner (CPU)..."
	cd examples/models/llama && cmake --workflow --preset llama-release
	@echo ""
	@echo "✓ Build complete!"
	@echo "  Binary: cmake-out/examples/models/llama/llama_main"

llava-cpu:
	@echo "==> Building and installing ExecuTorch..."
	cmake --workflow --preset llm-release
	@echo "==> Building Llava runner (CPU)..."
	cd examples/models/llava && cmake --workflow --preset llava
	@echo ""
	@echo "✓ Build complete!"
	@echo "  Binary: cmake-out/examples/models/llava/llava_main"

gemma3-cuda:
	@echo "==> Building and installing ExecuTorch with CUDA..."
	cmake --workflow --preset llm-release-cuda
	@echo "==> Building Gemma3 runner with CUDA..."
	cd examples/models/gemma3 && cmake --workflow --preset gemma3-cuda
	@echo ""
	@echo "✓ Build complete!"
	@echo "  Binary: cmake-out/examples/models/gemma3/gemma3_e2e_runner"

gemma3-cpu:
	@echo "==> Building and installing ExecuTorch..."
	cmake --workflow --preset llm-release
	@echo "==> Building Gemma3 runner (CPU)..."
	cd examples/models/gemma3 && cmake --workflow --preset gemma3-cpu
	@echo ""
	@echo "✓ Build complete!"
	@echo "  Binary: cmake-out/examples/models/gemma3/gemma3_e2e_runner"

executor-runner-xnnpack:
	@echo "==> Building generic executor_runner with XNNPack backend..."
	cmake --workflow --preset executor-runner-xnnpack
	@echo ""
	@echo "✓ Build complete!"
	@echo "  Binary: cmake-out/executor_runner"
	@echo ""
	@echo "Usage example:"
	@echo "  ./cmake-out/executor_runner --model_path=./xnnpack_model/model.pte"

clean:
	rm -rf cmake-out \
	       extension/llm/tokenizers/build \
	       extension/llm/tokenizers/pytorch_tokenizers.egg-info
