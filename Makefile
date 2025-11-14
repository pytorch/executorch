.PHONY: voxtral-cuda voxtral-cpu voxtral-metal whisper-cuda whisper-cpu whisper-metal llama-cpu llava-cpu gemma3-cuda gemma3-cpu clean help

help:
	@echo "Available targets:"
	@echo "  voxtral-cuda   - Build Voxtral runner with CUDA backend"
	@echo "  voxtral-cpu    - Build Voxtral runner with CPU backend"
	@echo "  voxtral-metal  - Build Voxtral runner with Metal backend (macOS only)"
	@echo "  whisper-cuda   - Build Whisper runner with CUDA backend"
	@echo "  whisper-cpu    - Build Whisper runner with CPU backend"
	@echo "  whisper-metal  - Build Whisper runner with Metal backend (macOS only)"
	@echo "  llama-cpu      - Build Llama runner with CPU backend"
	@echo "  llava-cpu      - Build Llava runner with CPU backend"
	@echo "  gemma3-cuda    - Build Gemma3 runner with CUDA backend"
	@echo "  gemma3-cpu     - Build Gemma3 runner with CPU backend"
	@echo "  clean          - Clean build artifacts"

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

clean:
	rm -rf cmake-out
