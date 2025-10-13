#!/bin/bash
# === CI Wheel Build & Test Script ===

# Exit immediately on error, print each command, and capture all output to build.log
set -e
set -x
exec > >(tee -i build.log) 2>&1

# Save repo root
REPO_ROOT=$(pwd)

# ----------------------------
# Dynamically create script_qnn_wheel_test.py
# ----------------------------
cat > "/tmp/script_qnn_wheel_test.py" << 'EOF'
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
from executorch.exir.backend.utils import format_delegated_graph
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
    output_graph = format_delegated_graph(delegated_program.exported_program().graph_module)
    # Ensure QnnBackend is in the output graph
    assert "QnnBackend" in output_graph
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
source .ci/scripts/utils.sh
install_executorch
EXECUTORCH_BUILDING_WHEEL=1 python setup.py bdist_wheel
unset EXECUTORCH_BUILDING_WHEEL

WHEEL_FILE=$(ls dist/*.whl | head -n 1)
echo "Found wheel: $WHEEL_FILE"

PYTHON_VERSION=$1
# ----------------------------
# Check wheel does NOT contain qualcomm/sdk
# ----------------------------
echo "Checking wheel does not contain qualcomm/sdk..."
SDK_FILES=$(unzip -l "$WHEEL_FILE" | awk '{print $4}' | grep -E "executorch/backends/qualcomm/sdk" || true)
if [ -n "$SDK_FILES" ]; then
    echo "ERROR: Wheel package contains unexpected qualcomm/sdk files:"
    echo "$SDK_FILES"
    exit 1
else
    echo "OK: No qualcomm/sdk files found in wheel"
fi

# ----------------------------
# Check .so files in the wheel
# ----------------------------
echo "Checking for .so files inside the wheel..."
WHEEL_SO_FILES=$(unzip -l "$WHEEL_FILE" | awk '{print $4}' | grep -E "executorch/backends/qualcomm/python" || true)
if [ -z "$WHEEL_SO_FILES" ]; then
    echo "ERROR: No .so files found in wheel under executorch/backends/qualcomm/python"
    exit 1
else
    echo "Wheel contains the following .so files:"
    echo "$WHEEL_SO_FILES"
fi

# ----------------------------
# Helpers
# ----------------------------
get_site_packages_dir () {
  local PYBIN="$1"
  "$PYBIN" - <<'PY'
import sysconfig, sys
print(sysconfig.get_paths().get("purelib") or sysconfig.get_paths().get("platlib"))
PY
}

run_core_tests () {
  local PYBIN="$1"      # path to python
  local PIPBIN="$2"     # path to pip
  local LABEL="$3"      # label to print (conda/venv)

  echo "=== [$LABEL] Installing wheel & deps ==="
  "$PIPBIN" install --upgrade pip
  "$PIPBIN" install "$WHEEL_FILE"
  TORCH_VERSION=$(
  "$PYBIN" - <<'PY'
import runpy
module_vars = runpy.run_path("torch_pin.py")
print(module_vars["TORCH_VERSION"])
PY
)

  NIGHTLY_VERSION=$(
  "$PYBIN" - <<'PY'
import runpy
module_vars = runpy.run_path("torch_pin.py")
print(module_vars["NIGHTLY_VERSION"])
PY
)
  echo "=== [$LABEL] Install torch==${TORCH_VERSION}.${NIGHTLY_VERSION} ==="

  # Install torchao based on the pinned PyTorch version
  "$PIPBIN" install torch=="${TORCH_VERSION}.${NIGHTLY_VERSION}" --index-url "https://download.pytorch.org/whl/nightly/cpu"
  "$PIPBIN" install wheel

  # Install torchao based on the pinned commit from third-party/ao submodule
  pushd "$REPO_ROOT/third-party/ao" > /dev/null
  export USE_CPP=0
  "$PIPBIN" install . --no-build-isolation
  popd > /dev/null

  echo "=== [$LABEL] Import smoke tests ==="
  "$PYBIN" -c "import executorch; print('executorch imported successfully')"
  "$PYBIN" -c "import executorch.backends.qualcomm; print('executorch.backends.qualcomm imported successfully')"
  "$PYBIN" -c "from executorch.export.target_recipes import get_android_recipe; recipe = get_android_recipe('android-arm64-snapdragon-fp16'); print(f'executorch.export.target_recipes imported successfully: {recipe}')"

  echo "=== [$LABEL] List installed executorch/backends/qualcomm/python ==="
  local SITE_DIR
  SITE_DIR="$(get_site_packages_dir "$PYBIN")"
  local SO_DIR="$SITE_DIR/executorch/backends/qualcomm/python"
  ls -l "$SO_DIR" || echo "Folder does not exist!"

  echo "=== [$LABEL] Run export script to generate linear.pte ==="
  (cd "$REPO_ROOT" && "$PYBIN" "/tmp/script_qnn_wheel_test.py")

  if [ -f "$REPO_ROOT/linear.pte" ]; then
      echo "[$LABEL] Model file linear.pte successfully created"
  else
      echo "ERROR: [$LABEL] Model file linear.pte was not created"
      exit 1
  fi
}

# ----------------------------
# Conda environment setup & tests
# ----------------------------
echo "=== Testing in Conda env ==="
TEMP_ENV_DIR=$(mktemp -d)
echo "Using temporary directory for conda: $TEMP_ENV_DIR"
conda create -y -p "$TEMP_ENV_DIR/env" python=$PYTHON_VERSION
# derive python/pip paths inside the conda env
CONDA_PY="$TEMP_ENV_DIR/env/bin/python"
CONDA_PIP="$TEMP_ENV_DIR/env/bin/pip"
# Some images require conda run; keep pip/python direct to simplify path math
run_core_tests "$CONDA_PY" "$CONDA_PIP" "conda"

# Cleanup conda env
conda env remove -p "$TEMP_ENV_DIR/env" -y || true
rm -rf "$TEMP_ENV_DIR"

# ----------------------------
# Python venv setup & tests
# ----------------------------
echo "=== Testing in Python venv ==="
TEMP_VENV_DIR=$(mktemp -d)
echo "Using temporary directory for venv: $TEMP_VENV_DIR"
python3 -m venv "$TEMP_VENV_DIR/venv"
VENV_PY="$TEMP_VENV_DIR/venv/bin/python"
VENV_PIP="$TEMP_VENV_DIR/venv/bin/pip"

# Ensure venv has wheel/build basics if needed
"$VENV_PIP" install --upgrade pip

run_core_tests "$VENV_PY" "$VENV_PIP" "venv"

# Cleanup venv
rm -rf "$TEMP_VENV_DIR"

echo "=== All tests completed! ==="
