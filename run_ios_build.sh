bash install_executorch.sh --clean

bash install_executorch.sh

pip install -e .

./install_requirements.sh

# CoreML-only requirements:
./backends/apple/coreml/scripts/install_requirements.sh

# MPS-only requirements:
./backends/apple/mps/install_requirements.sh

./scripts/build_apple_frameworks.sh --Release --coreml --mps --xnnpack --custom --optimized --quantized --torchao
