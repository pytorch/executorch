#!/bin/bash

set -e  # Exit on any error

echo "=== Building Wheel Package ==="
python setup.py bdist_wheel

# Find the wheel file
WHEEL_FILE=$(ls dist/*.whl | head -n 1)
echo "Found wheel: $WHEEL_FILE"

echo "=== Checking for expected .so files ==="
SO_FILES=$(unzip -l "$WHEEL_FILE" | grep "\.so" | grep "qualcomm")

echo "=== Checking for expected .so files ==="
echo "Listing all .so files in the wheel:"
unzip -l "$WHEEL_FILE" | grep "\.so" || echo "No .so files found in the wheel!"

# List all .so files with full paths
ALL_SO_FILES=$(unzip -l "$WHEEL_FILE" | awk '{print $4}' | grep "\.so")

echo "All .so files in wheel:"
echo "$ALL_SO_FILES"

# Define expected .so files
EXPECTED_SO_FILES=(
    "executorch/backends/qualcomm/qnn_backend.cpython-310-x86_64-linux-gnu.so"
    "executorch/backends/qualcomm/python/PyQnnManagerAdaptor.cpython-310-x86_64-linux-gnu.so"
    "executorch/backends/qualcomm/python/PyQnnWrapperAdaptor.cpython-310-x86_64-linux-gnu.so"
)

MISSING=false
for file in "${EXPECTED_SO_FILES[@]}"; do
    if echo "$ALL_SO_FILES" | grep -q "$file"; then
        echo "Found expected .so file: $file"
    else
        echo "ERROR: Missing expected .so file: $file"
        MISSING=true
    fi
done

if [ "$MISSING" = true ]; then
    echo "==== .so file check failed ===="
    echo "Wheel contents for debugging:"
    unzip -l "$WHEEL_FILE"
    exit 1
else
    echo "All expected .so files found in wheel ✅"
fi

# Create a temporary directory for our test environment
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
    exit 1
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
    exit 1
fi

echo "=== Running model generation script ==="
conda run -p "$TEMP_ENV_DIR/env" python script.py

# Check if linear.pte file was created
if [ -f "linear.pte" ]; then
    echo "Model file linear.pte successfully created"
else
    echo "ERROR: Model file linear.pte was not created"
    exit 1
fi

echo "=== Cleaning up ==="
conda env remove -p "$TEMP_ENV_DIR/env" -y
rm -rf "$TEMP_ENV_DIR"

echo "=== All tests passed! ==="
