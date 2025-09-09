#!/bin/bash
# === CI Wheel Build & Test Script ===

# Exit immediately on error, print each command, and capture all output to build.log
set -e
set -x
exec > >(tee -i build.log) 2>&1

echo "=== Building Wheel Package ==="
python setup.py bdist_wheel

# Find the wheel file
WHEEL_FILE=$(ls dist/*.whl | head -n 1)
echo "Found wheel: $WHEEL_FILE"

echo "=== Checking for expected .so files ==="

# List all .so files in the wheel
echo "Listing all .so files in the wheel:"
ALL_SO_FILES=$(unzip -l "$WHEEL_FILE" | awk '{print $4}' | grep "\.so" || true)

if [ -z "$ALL_SO_FILES" ]; then
    echo "WARNING: No .so files found in the wheel!"
else
    echo "$ALL_SO_FILES"
fi

# Define expected .so files
EXPECTED_SO_FILES=(
    "executorch/backends/qualcomm/qnn_backend.cpython-310-x86_64-linux-gnu.so"
    "executorch/backends/qualcomm/python/PyQnnManagerAdaptor.cpython-310-x86_64-linux-gnu.so"
    "executorch/backends/qualcomm/python/PyQnnWrapperAdaptor.cpython-310-x86_64-linux-gnu.so"
)

# Check each expected .so file
MISSING=false
for file in "${EXPECTED_SO_FILES[@]}"; do
    if echo "$ALL_SO_FILES" | grep -q "$file"; then
        echo "Found expected .so file: $file"
    else
        echo "ERROR: Missing expected .so file: $file"
        MISSING=true
    fi
done

# Print wheel contents if missing any .so
if [ "$MISSING" = true ]; then
    echo "==== .so file check failed ===="
    echo "Full wheel contents for debugging:"
    unzip -l "$WHEEL_FILE"
    # Continue for debugging instead of exiting immediately
    # exit 1
fi

# Create a temporary directory for test environment
TEMP_ENV_DIR=$(mktemp -d)
echo "Using temporary directory: $TEMP_ENV_DIR"

echo "=== Creating and testing in conda environment ==="
conda create -y -p "$TEMP_ENV_DIR/env" python=3.10
conda run -p "$TEMP_ENV_DIR/env" pip install "$WHEEL_FILE"

echo "=== Testing import without SDK download ==="
conda run -p "$TEMP_ENV_DIR/env" python -c "import executorch; print('executorch imported successfully')"

# Check that SDK directory doesn't exist after first import
SDK_PATH="$TEMP_ENV_DIR/env/lib/python3.10/site-packages/executorch/backends/qualcomm/sdk"
if [ -d "$SDK_PATH" ]; then
    echo "ERROR: SDK directory exists after first import: $SDK_PATH"
else
    echo "SDK directory correctly doesn't exist after first import"
fi

echo "=== Testing import that should trigger SDK download ==="
conda run -p "$TEMP_ENV_DIR/env" python -c "import executorch.backends.qualcomm; print('executorch.backends.qualcomm imported successfully')"

# Check that SDK directory exists after second import
if [ -d "$SDK_PATH" ]; then
    echo "SDK directory correctly exists after second import: $SDK_PATH"
else
    echo "ERROR: SDK directory doesn't exist after second import"
fi

echo "=== Running model generation script ==="
conda run -p "$TEMP_ENV_DIR/env" python script.py

# Check if linear.pte file was created
if [ -f "linear.pte" ]; then
    echo "Model file linear.pte successfully created"
else
    echo "ERROR: Model file linear.pte was not created"
fi

echo "=== Cleaning up ==="
conda env remove -p "$TEMP_ENV_DIR/env" -y
rm -rf "$TEMP_ENV_DIR"

echo "=== All tests completed! ==="
