#!/bin/bash
# === OpenVINO Wheel Build & Test Script ===
#
# Builds the ExecuTorch wheel with OpenVINO support, installs it into both
# a conda env and a Python venv, and runs smoke tests verifying that the
# OpenVINO backend is registered and can export a simple model.


set -e
set -x
exec > >(tee -i openvino_wheel_build.log) 2>&1

REPO_ROOT=$(pwd)
PYTHON_VERSION=${1:-3.11}

# ----------------------------
# Dynamically create test script
# ----------------------------
cat > "/tmp/script_openvino_wheel_test.py" << 'EOF'
import torch
from torch.export import export
from executorch.backends.openvino.partitioner import OpenvinoPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.backend.utils import format_delegated_graph
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.extension.export_util.utils import save_pte_program


class LinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x):
        return self.linear(x)

    def get_example_inputs(self):
        return (torch.randn(3, 3),)


model = LinearModule().eval()
example_inputs = model.get_example_inputs()

exported = export(model, example_inputs, strict=True)

compile_spec = [CompileSpec("device", b"CPU")]
edge = to_edge_transform_and_lower(
    exported,
    partitioner=[OpenvinoPartitioner(compile_spec)],
)

# Verify OpenVINO delegation occurred
output_graph = format_delegated_graph(edge.exported_program().graph_module)
assert "OpenvinoBackend" in output_graph, \
    "Expected OpenVINO delegation but no delegate call found in graph"
print("OpenVINO delegation successful")

executorch_program = edge.to_executorch(
    config=ExecutorchBackendConfig(extract_delegate_segments=False)
)
save_pte_program(executorch_program, "linear_openvino", "")
print("linear_openvino.pte created successfully")
EOF

source .ci/scripts/utils.sh
# ----------------------------
# Install OpenVINO and source its setupvars.sh
# ----------------------------
echo "=== Installing OpenVINO ==="
source "${REPO_ROOT}/backends/openvino/scripts/install_openvino.sh"
install_openvino
echo "OpenVINO_DIR=${OpenVINO_DIR}"

# ----------------------------
# Build the wheel
# ----------------------------
echo "=== Building Wheel Package ==="
install_executorch

python setup.py bdist_wheel

WHEEL_FILE=$(ls dist/*.whl | head -n 1)
echo "Found wheel: ${WHEEL_FILE}"


# ----------------------------
# Helpers
# ----------------------------
run_core_tests() {
    local PYBIN="$1"
    local PIPBIN="$2"
    local LABEL="$3"

    echo "=== [${LABEL}] Installing wheel ==="
    "${PIPBIN}" install --upgrade pip
    "${PIPBIN}" install "${WHEEL_FILE}"

    echo "=== [${LABEL}] Import smoke tests ==="
    "${PYBIN}" -c "import executorch; print('executorch imported')"
    "${PYBIN}" -c "import executorch.backends.openvino; print('executorch.backends.openvino imported')"

    echo "=== [${LABEL}] Verify OpenvinoBackend is registered ==="
    "${PYBIN}" - <<'PY'
from executorch.extension.pybindings.portable_lib import _get_registered_backend_names
backends = _get_registered_backend_names()
print(f"Registered backends: {backends}")
assert "OpenvinoBackend" in backends, \
    f"OpenvinoBackend not found in registered backends: {backends}"
print("OpenvinoBackend is registered")
PY

    echo "=== [${LABEL}] Run export script to generate linear_openvino.pte ==="
    (cd "${REPO_ROOT}" && "${PYBIN}" /tmp/script_openvino_wheel_test.py)

    if [[ -f "${REPO_ROOT}/linear_openvino.pte" ]]; then
        echo "[${LABEL}] linear_openvino.pte created successfully"
        rm -f "${REPO_ROOT}/linear_openvino.pte"
    else
        echo "ERROR: [${LABEL}] linear_openvino.pte was not created"
        exit 1
    fi
}

# ----------------------------
# Conda environment tests
# ----------------------------
echo "=== Testing in Conda env ==="
TEMP_ENV_DIR=$(mktemp -d)
echo "Using temporary directory for conda: $TEMP_ENV_DIR"
conda create -y -p "${TEMP_ENV_DIR}/env" python="${PYTHON_VERSION}"
CONDA_PY="${TEMP_ENV_DIR}/env/bin/python"
CONDA_PIP="${TEMP_ENV_DIR}/env/bin/pip"
run_core_tests "${CONDA_PY}" "${CONDA_PIP}" "conda"
conda env remove -p "${TEMP_ENV_DIR}/env" -y || true
rm -rf "${TEMP_ENV_DIR}"

# ----------------------------
# Python venv tests
# ----------------------------
echo "=== Testing in Python venv ==="
TEMP_VENV_DIR=$(mktemp -d)
python3 -m venv "${TEMP_VENV_DIR}/venv"
VENV_PY="${TEMP_VENV_DIR}/venv/bin/python"
VENV_PIP="${TEMP_VENV_DIR}/venv/bin/pip"
run_core_tests "${VENV_PY}" "${VENV_PIP}" "venv"
rm -rf "${TEMP_VENV_DIR}"

echo "=== All OpenVINO wheel tests completed! ==="
