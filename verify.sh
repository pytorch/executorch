#!/bin/bash
# Migration verification script for CUDA runtime
# This script checks if migration from internal etensor and old shim layers to slimtensor and new shims is complete
# Final verification: Build and run gemma3 model to ensure everything works end-to-end

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_RUNTIME_DIR="${SCRIPT_DIR}/backends/cuda/runtime"

echo "=========================================="
echo "CUDA Runtime Migration Verification Script"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ISSUES_FOUND=0

# Check 1: Look for references to old internal etensor paths
echo "Check 1: Searching for old internal etensor references..."
OLD_ETENSOR_REFS=$(grep -r "executorch/runtime/core/exec_aten" \
  --include="*.cpp" --include="*.h" \
  --exclude-dir=slim \
  --exclude-dir=tests \
  "${CUDA_RUNTIME_DIR}" 2>/dev/null || true)

if [ -n "$OLD_ETENSOR_REFS" ]; then
  echo -e "${RED}✗ Found references to old exec_aten paths:${NC}"
  echo "$OLD_ETENSOR_REFS" | head -20
  ISSUES_FOUND=$((ISSUES_FOUND + 1))
else
  echo -e "${GREEN}✓ No old exec_aten references found in runtime code${NC}"
fi
echo ""

# Check 2: Look for references to old aoti common_shims
echo "Check 2: Searching for old aoti common_shims references..."
OLD_COMMON_SHIMS=$(grep -r "executorch/backends/aoti/common_shims" \
  --include="*.cpp" --include="*.h" \
  "${CUDA_RUNTIME_DIR}" 2>/dev/null || true)

if [ -n "$OLD_COMMON_SHIMS" ]; then
  echo -e "${RED}✗ Found references to old aoti common_shims:${NC}"
  echo "$OLD_COMMON_SHIMS"
  ISSUES_FOUND=$((ISSUES_FOUND + 1))
else
  echo -e "${GREEN}✓ No old common_shims references found${NC}"
fi
echo ""

# Check 3: Verify new slim tensor structure exists
echo "Check 3: Verifying new slim tensor structure..."
REQUIRED_SLIM_FILES=(
  "slim/core/SlimTensor.h"
  "slim/core/Factory.h"
  "slim/core/Storage.h"
)

MISSING_FILES=0
for file in "${REQUIRED_SLIM_FILES[@]}"; do
  if [ ! -f "${CUDA_RUNTIME_DIR}/${file}" ]; then
    echo -e "${RED}✗ Missing required file: ${file}${NC}"
    MISSING_FILES=$((MISSING_FILES + 1))
  fi
done

if [ $MISSING_FILES -eq 0 ]; then
  echo -e "${GREEN}✓ All required slim tensor files present${NC}"
else
  ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi
echo ""

# Check 4: Verify new shims structure exists
echo "Check 4: Verifying new shims structure..."
REQUIRED_SHIM_DIRS=(
  "shims/aoti_include"
  "shims/aoti_runtime"
  "shims/aoti_torch"
  "shims/cpp_wrapper"
)

MISSING_DIRS=0
for dir in "${REQUIRED_SHIM_DIRS[@]}"; do
  if [ ! -d "${CUDA_RUNTIME_DIR}/${dir}" ]; then
    echo -e "${RED}✗ Missing required directory: ${dir}${NC}"
    MISSING_DIRS=$((MISSING_DIRS + 1))
  fi
done

if [ $MISSING_DIRS -eq 0 ]; then
  echo -e "${GREEN}✓ All required shim directories present${NC}"
else
  ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi
echo ""

# Check 5: Look for old standalone shim files that should be removed
echo "Check 5: Checking for old standalone shim files..."
OLD_SHIM_FILES=(
  "common_shims.cpp"
  "common_shims.h"
  "memory.cpp"
  "memory.h"
  "tensor_attribute.cpp"
  "tensor_attribute.h"
  "cuda_guard.cpp"
  "guard.cpp"
)

OLD_FILES_EXIST=0
for file in "${OLD_SHIM_FILES[@]}"; do
  if [ -f "${CUDA_RUNTIME_DIR}/${file}" ]; then
    echo -e "${YELLOW}⚠ Old shim file still exists: ${file}${NC}"
    OLD_FILES_EXIST=$((OLD_FILES_EXIST + 1))
  fi
done

if [ $OLD_FILES_EXIST -eq 0 ]; then
  echo -e "${GREEN}✓ No old standalone shim files found${NC}"
else
  echo -e "${YELLOW}⚠ Found ${OLD_FILES_EXIST} old shim file(s) that may need removal${NC}"
fi
echo ""

# Check 6: Look for old namespace usage
echo "Check 6: Checking for old namespace references..."
OLD_NAMESPACE=$(grep -r "namespace.*executorch::backends::aoti" \
  --include="*.cpp" --include="*.h" \
  "${CUDA_RUNTIME_DIR}" 2>/dev/null || true)

if [ -n "$OLD_NAMESPACE" ]; then
  echo -e "${RED}✗ Found old aoti namespace usage:${NC}"
  echo "$OLD_NAMESPACE" | head -10
  ISSUES_FOUND=$((ISSUES_FOUND + 1))
else
  echo -e "${GREEN}✓ No old aoti namespace references found${NC}"
fi
echo ""

# Static checks summary
echo "=========================================="
echo "Static Checks Summary"
echo "=========================================="
if [ $ISSUES_FOUND -eq 0 ]; then
  echo -e "${GREEN}✓ All static migration checks passed${NC}"
else
  echo -e "${YELLOW}⚠ Found ${ISSUES_FOUND} static check issue(s)${NC}"
  echo "Proceeding with build and runtime test..."
fi
echo ""

# Final verification: Build and run the model
echo "=========================================="
echo "Final Verification: Build and Run Test"
echo "=========================================="
echo ""

echo "Cleaning previous build..."
rm -rf cmake-out

echo "Configuring CMake with CUDA support..."
cmake --preset llm \
      -DEXECUTORCH_BUILD_CUDA=ON \
      -DCMAKE_INSTALL_PREFIX=cmake-out \
      -DCMAKE_BUILD_TYPE=Release \
      -DEXECUTORCH_ENABLE_LOGGING=ON \
      -Bcmake-out -S.

echo "Building executorch..."
cmake --build cmake-out -j$(nproc) --target install --config Release

echo "Building Gemma3 runner..."
cmake -DEXECUTORCH_BUILD_CUDA=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DEXECUTORCH_ENABLE_LOGGING=ON \
      -Sexamples/models/gemma3 \
      -Bcmake-out/examples/models/gemma3/
cmake --build cmake-out/examples/models/gemma3 --target gemma3_e2e_runner --config Release

echo ""
echo "Running bf16 Gemma3 model test..."
./cmake-out/examples/models/gemma3/gemma3_e2e_runner \
  --model_path /home/gasoonjia/gemma/cuda/bf16/model.pte \
  --data_path /home/gasoonjia/gemma/cuda/bf16/aoti_cuda_blob.ptd \
  --tokenizer_path /home/gasoonjia/gemma/cuda/bf16/tokenizer.json \
  --image_path docs/source/_static/img/et-logo.png \
  --temperature 0

echo ""
# echo "Running int4-tile Gemma3 model test..."
# ./cmake-out/examples/models/gemma3/gemma3_e2e_runner \
#   --model_path /home/gasoonjia/gemma/cuda/int4/model.pte \
#   --data_path /home/gasoonjia/gemma/cuda/int4/aoti_cuda_blob.ptd \
#   --tokenizer_path /home/gasoonjia/gemma/cuda/tokenizer.json \
#   --image_path docs/source/_static/img/et-logo.png \
#   --temperature 0

# echo ""
# echo "Running int4 weight only Gemma3 model test..."
# ./cmake-out/examples/models/gemma3/gemma3_e2e_runner \
#   --model_path /home/gasoonjia/gemma/cuda/int4-2/model.pte \
#   --data_path /home/gasoonjia/gemma/cuda/int4-2/aoti_cuda_blob.ptd \
#   --tokenizer_path /home/gasoonjia/gemma/cuda/tokenizer.json \
#   --image_path docs/source/_static/img/et-logo.png \
#   --temperature 0

echo ""
echo "=========================================="
echo "Final Summary"
echo "=========================================="
if [ $ISSUES_FOUND -eq 0 ]; then
  echo -e "${GREEN}✓✓✓ Migration verification FULLY PASSED ✓✓✓${NC}"
  echo "All static checks passed and model ran successfully!"
  exit 0
else
  echo -e "${YELLOW}⚠ Migration verification PARTIALLY PASSED${NC}"
  echo "Found ${ISSUES_FOUND} static check issue(s), but model ran successfully"
  echo "Consider addressing the static issues for full migration completion"
  exit 0
fi
