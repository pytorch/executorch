#!/bin/bash
# === CI Wheel Build & Test Script ===

# Exit immediately on error, print each command, and capture all output to build.log
set -e
set -x
exec > >(tee -i build.log) 2>&1

# Save repo root
REPO_ROOT=$(pwd)

# ----------------------------
# Dynamically create script.py
# ----------------------------
cat > "$REPO_ROOT/script.py" << 'EOF'
# pyre-ignore-all-errors
import argparse

import torch
from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.examples.models.model_factory import EagerModelFactory
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.extension.export_util.utils import save_pte_program
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e, prepare_qat_pt2e

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--output_folder", type=str, default="", help="The folder to store the exported program")
    parser.add_argument("--soc", type=str, default="SM8650", help="Specify the SoC model.")
    parser.add_argument("-q", "--quantization", choices=["ptq", "qat"], help="Run post-traininig quantization.")
    args = parser.parse_args()

    class LinearModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)
        def forward(self, arg):
            return self.linear(arg)
        def get_example_inputs(self):
            return (torch.randn(3, 3),)

    model = LinearModule()
    example_inputs = model.get_example_inputs()

    if args.quantization:
        quantizer = QnnQuantizer()
        m = torch.export.export(model.eval(), example_inputs, strict=True).module()
        if args.quantization == "qat":
            m = prepare_qat_pt2e(m, quantizer)
            m(*example_inputs)
        elif args.quantization == "ptq":
            m = prepare_pt2e(m, quantizer)
            m(*example_inputs)
        m = convert_pt2e(m)
    else:
        m = model

    use_fp16 = True if args.quantization is None else False
    backend_options = generate_htp_compiler_spec(use_fp16=use_fp16)
    compile_spec = generate_qnn_executorch_compiler_spec(
        soc_model=get_soc_to_chipset_map()[args.soc],
        backend_options=backend_options,
    )
    delegated_program = to_edge_transform_and_lower_to_qnn(m, example_inputs, compile_spec)
    executorch_program = delegated_program.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )
    save_pte_program(executorch_program, "linear", args.output_folder)

if __name__ == "__main__":
    main()
EOF

# ----------------------------
# Wheel build and .so checks
# ----------------------------
echo "=== Building Wheel Package ==="
python setup.py bdist_wheel

WHEEL_FILE=$(ls dist/*.whl | head -n 1)
echo "Found wheel: $WHEEL_FILE"

# (Keep all your existing .so checks here...)

# ----------------------------
# Conda environment setup & tests
# ----------------------------
TEMP_ENV_DIR=$(mktemp -d)
echo "Using temporary directory: $TEMP_ENV_DIR"

conda create -y -p "$TEMP_ENV_DIR/env" python=3.10
conda run -p "$TEMP_ENV_DIR/env" pip install "$WHEEL_FILE"

conda run -p "$TEMP_ENV_DIR/env" pip install torch=="2.9.0.dev20250801" --index-url "https://download.pytorch.org/whl/nightly/cpu"
conda run -p "$TEMP_ENV_DIR/env" pip install --pre torchao --index-url "https://download.pytorch.org/whl/nightly/cpu"

# ----------------------------
# Check .so files in the wheel
# ----------------------------
echo "Checking for .so files inside the wheel..."
WHEEL_SO_FILES=$(unzip -l "$WHEEL_FILE" | awk '{print $4}' | grep "executorch/backends/qualcomm/python" || true)
if [ -z "$WHEEL_SO_FILES" ]; then
    echo "WARNING: No .so files found in wheel under executorch/backends/qualcomm/python"
else
    echo "Wheel contains the following .so files:"
    echo "$WHEEL_SO_FILES"
fi

# ----------------------------
# Check installed .so files in conda env
# ----------------------------
echo "Checking installed executorch/backends/qualcomm/python contents in conda env..."
ENV_SO_DIR="$TEMP_ENV_DIR/env/lib/python3.10/site-packages/executorch/backends/qualcomm/python"
ls -l "$ENV_SO_DIR" || echo "Folder does not exist!"

# Run import tests
conda run -p "$TEMP_ENV_DIR/env" python -c "import executorch; print('executorch imported successfully')"
conda run -p "$TEMP_ENV_DIR/env" python -c "import executorch.backends.qualcomm; print('executorch.backends.qualcomm imported successfully')"

# Run the dynamically created script using absolute path
conda run -p "$TEMP_ENV_DIR/env" python "$REPO_ROOT/script.py"

# Check if linear.pte was created
if [ -f "linear.pte" ]; then
    echo "Model file linear.pte successfully created"
else
    echo "ERROR: Model file linear.pte was not created"
fi

# Cleanup
conda env remove -p "$TEMP_ENV_DIR/env" -y
rm -rf "$TEMP_ENV_DIR"

echo "=== All tests completed! ==="
