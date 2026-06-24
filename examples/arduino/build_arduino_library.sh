#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# build_arduino_library.sh — Assemble the ExecuTorch Arduino library
# from sources in this repository.
#
# Usage:
#   ./build_arduino_library.sh                # build the library
#   ./build_arduino_library.sh --clean        # remove generated output
#   ./build_arduino_library.sh --bump patch   # 0.1.0 → 0.1.1
#   ./build_arduino_library.sh --bump minor   # 0.1.0 → 0.2.0
#   ./build_arduino_library.sh --bump major   # 0.1.0 → 1.0.0
#
# Output: arduino_lib/ExecuTorchArduino/ (self-contained, installable)
#
# NOTE: This script is coupled to the ExecuTorch source tree layout.
# Long-term, we should use cmake query APIs to deduce required sources
# for a given target. Short-term, a CI smoke test will catch breakage.
# When we set up the separate pytorch/executorch-arduino repo for
# Library Manager publishing, this script may move there with ET as a
# submodule.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ET_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR="$SCRIPT_DIR/arduino_lib/ExecuTorchArduino"
PROPS="$SCRIPT_DIR/library.properties"

if [ "${1:-}" = "--clean" ]; then
  echo "Cleaning generated library..."
  rm -rf "$SCRIPT_DIR/arduino_lib"
  echo "Done."
  exit 0
fi

if [ "${1:-}" = "--bump" ]; then
  PART="${2:-patch}"
  CURRENT=$(grep "^version=" "$PROPS" | cut -d= -f2)
  IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"
  case "$PART" in
    major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
    minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
    patch) PATCH=$((PATCH + 1)) ;;
    *) echo "Usage: $0 --bump [major|minor|patch]"; exit 1 ;;
  esac
  NEW="$MAJOR.$MINOR.$PATCH"
  sed -i '' "s/^version=.*/version=$NEW/" "$PROPS" 2>/dev/null || \
    sed -i "s/^version=.*/version=$NEW/" "$PROPS"
  echo "Version: $CURRENT → $NEW"
  exit 0
fi

echo "=== Building ExecuTorch Arduino Library ==="
echo "  ET repo:  $ET_ROOT"
echo "  Output:   $OUT_DIR"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/src" "$OUT_DIR/examples"

# ─────────────────────────────────────────────────────────
# 1. Copy library metadata, wrapper header, and stubs
# ─────────────────────────────────────────────────────────
cp "$SCRIPT_DIR/library.properties" "$OUT_DIR/"
cp "$SCRIPT_DIR/ExecuTorchArduino.h" "$OUT_DIR/src/"
cp "$SCRIPT_DIR/platform_stubs.c" "$OUT_DIR/src/"
cp -r "$SCRIPT_DIR/examples/"* "$OUT_DIR/examples/"

echo "[1/7] Metadata and examples copied"

# ─────────────────────────────────────────────────────────
# 2. Vendor ET runtime sources
# ─────────────────────────────────────────────────────────
ET_SRC="$OUT_DIR/src/executorch"
mkdir -p "$ET_SRC"

for dir in runtime/core runtime/executor runtime/kernel \
           runtime/platform runtime/backend; do
  mkdir -p "$ET_SRC/$dir"
  find "$ET_ROOT/$dir" -maxdepth 1 -name "*.h" -exec cp {} "$ET_SRC/$dir/" \;
  find "$ET_ROOT/$dir" -maxdepth 1 -name "*.cpp" -exec cp {} "$ET_SRC/$dir/" \;
done

# Subdirectories with headers
for dir in runtime/core/exec_aten runtime/core/exec_aten/util \
           runtime/core/portable_type runtime/platform/default; do
  mkdir -p "$ET_SRC/$dir"
  find "$ET_ROOT/$dir" -maxdepth 1 -name "*.h" -exec cp {} "$ET_SRC/$dir/" \;
  find "$ET_ROOT/$dir" -maxdepth 1 -name "*.cpp" -exec cp {} "$ET_SRC/$dir/" \;
done

# Extension data loader (header-only)
mkdir -p "$ET_SRC/extension/data_loader"
cp "$ET_ROOT/extension/data_loader/"*.h "$ET_SRC/extension/data_loader/"

# Runner util headers
mkdir -p "$ET_SRC/extension/runner_util"
cp "$ET_ROOT/extension/runner_util/"*.h "$ET_SRC/extension/runner_util/" 2>/dev/null || true

# Schema headers (generated — need a prior cmake build)
mkdir -p "$ET_SRC/schema"
cp "$ET_ROOT/schema/"*.h "$ET_SRC/schema/" 2>/dev/null || true
# Look for generated headers in common build dirs
for build_dir in "$ET_ROOT/cmake-out" "$ET_ROOT/cmake-out-mac" \
                 "$ET_ROOT/outputs/build_uno_q"; do
  if [ -d "$build_dir/schema/include/executorch/schema" ]; then
    cp "$build_dir/schema/include/executorch/schema/"*.h "$ET_SRC/schema/"
    break
  fi
done

# Verify schema headers were found
if [ ! -f "$ET_SRC/schema/program_generated.h" ]; then
  echo "ERROR: Schema headers not found. Run a cmake build first (e.g. cmake -Bbuild -DCMAKE_INSTALL_PREFIX=cmake-out)."
  exit 1
fi

echo "[2/7] ET runtime sources copied"

# ─────────────────────────────────────────────────────────
# 3. Vendor portable kernels
# ─────────────────────────────────────────────────────────
mkdir -p "$ET_SRC/kernels/portable/cpu/util" \
         "$ET_SRC/kernels/portable/cpu/pattern"

# Copy all portable op sources and headers
find "$ET_ROOT/kernels/portable/cpu" -maxdepth 1 \( -name "*.cpp" -o -name "*.h" \) \
  -exec cp {} "$ET_SRC/kernels/portable/cpu/" \;
cp "$ET_ROOT/kernels/portable/cpu/util/"*.h "$ET_SRC/kernels/portable/cpu/util/"
cp "$ET_ROOT/kernels/portable/cpu/util/"*.cpp "$ET_SRC/kernels/portable/cpu/util/" 2>/dev/null || true
cp "$ET_ROOT/kernels/portable/cpu/pattern/"*.h "$ET_SRC/kernels/portable/cpu/pattern/"
cp "$ET_ROOT/kernels/portable/cpu/pattern/"*.cpp "$ET_SRC/kernels/portable/cpu/pattern/" 2>/dev/null || true

echo "[3/7] Portable kernels copied"

# ─────────────────────────────────────────────────────────
# 4. Vendor Cortex-M backend ops
# ─────────────────────────────────────────────────────────
mkdir -p "$ET_SRC/backends/cortex_m/ops"
cp "$ET_ROOT/backends/cortex_m/ops/"*.cpp "$ET_SRC/backends/cortex_m/ops/"
cp "$ET_ROOT/backends/cortex_m/ops/"*.h "$ET_SRC/backends/cortex_m/ops/"

echo "[4/7] Cortex-M ops copied"

# ─────────────────────────────────────────────────────────
# 5. Vendor third-party dependencies
# ─────────────────────────────────────────────────────────

# c10 / torch headers
cp -r "$ET_ROOT/runtime/core/portable_type/c10/c10" "$OUT_DIR/src/c10"
cp -r "$ET_ROOT/runtime/core/portable_type/c10/torch" "$OUT_DIR/src/torch"

# cmake_macros.h stub
mkdir -p "$OUT_DIR/src/torch/headeronly/macros"
cat > "$OUT_DIR/src/torch/headeronly/macros/cmake_macros.h" << 'STUB'
#pragma once
#define C10_BUILD_SHARED_LIBS
#define C10_USE_GLOG 0
#define C10_USE_MINIMAL_GLOG 0
#define C10_USE_GFLAGS 0
STUB

# flatcc runtime and headers
mkdir -p "$OUT_DIR/src/flatcc/portable"
cp "$ET_ROOT/third-party/flatcc/include/flatcc/"*.h "$OUT_DIR/src/flatcc/"
cp -r "$ET_ROOT/third-party/flatcc/include/flatcc/portable" "$OUT_DIR/src/flatcc/"
mkdir -p "$OUT_DIR/src/flatcc/runtime"
cp "$ET_ROOT/third-party/flatcc/src/runtime/"*.c "$OUT_DIR/src/flatcc/runtime/"

# flatbuffers headers
cp -r "$ET_ROOT/third-party/flatbuffers/include/flatbuffers" "$OUT_DIR/src/flatbuffers"

# CMSIS-NN (from Zephyr workspace or cmake fetchcontent)
CMSIS_NN=""
for candidate in \
  "$ET_ROOT/outputs/zephyrproject/modules/lib/cmsis-nn" \
  "$ET_ROOT/third-party/cmsis-nn" \
  "$ET_ROOT/backends/arm/third-party/cmsis-nn/CMSIS-NN"; do
  if [ -d "$candidate/Source" ]; then
    CMSIS_NN="$candidate"
    break
  fi
done

if [ -n "$CMSIS_NN" ]; then
  mkdir -p "$OUT_DIR/src/cmsis-nn"
  cp -r "$CMSIS_NN/Source" "$OUT_DIR/src/cmsis-nn/"
  cp "$CMSIS_NN/Include/"*.h "$OUT_DIR/src/" 2>/dev/null || true
  if [ -d "$CMSIS_NN/Include/Internal" ]; then
    mkdir -p "$OUT_DIR/src/Internal"
    cp "$CMSIS_NN/Include/Internal/"*.h "$OUT_DIR/src/Internal/"
  fi
  echo "[5/7] CMSIS-NN copied from $CMSIS_NN"
else
  echo "[5/7] WARNING: CMSIS-NN not found. Cortex-M ops will not link."
fi

# CMSIS Core headers (for arm_math_types.h)
for candidate in \
  "$ET_ROOT/outputs/zephyrproject/modules/hal/cmsis_6/CMSIS/Core/Include" \
  "$ET_ROOT/third-party/cmsis/CMSIS/Core/Include"; do
  if [ -d "$candidate" ]; then
    cp "$candidate/"*.h "$OUT_DIR/src/" 2>/dev/null || true
    break
  fi
done

echo "[6/7] Third-party dependencies copied"

# ─────────────────────────────────────────────────────────
# 6. Apply Arduino-specific patches
# ─────────────────────────────────────────────────────────

# Fix: #include <exception> before <variant> in all ET headers
find "$OUT_DIR/src/executorch" -name "*.h" -print0 | \
  xargs -0 perl -pi -e 's/#include <variant>/#include <exception>\n#include <variant>/g'

# Remove test files, ATen-specific files, non-Zephyr platform backends
find "$OUT_DIR" -path "*testing*" -delete 2>/dev/null || true
find "$OUT_DIR" -name "*_aten.cpp" -delete 2>/dev/null || true
find "$OUT_DIR" -path "*test*" -name "*.cpp" -delete 2>/dev/null || true
rm -f "$OUT_DIR/src/executorch/runtime/platform/default/android.cpp"
rm -f "$OUT_DIR/src/executorch/runtime/platform/default/posix.cpp"
rm -f "$OUT_DIR/src/executorch/runtime/platform/default/windows.cpp"

# Regenerate schema headers if flatc is available
FLATC=""
for flatc_candidate in "$ET_ROOT/cmake-out-mac/third-party/flatc_ep/bin/flatc" \
                      "$ET_ROOT/cmake-out/third-party/flatc_ep/bin/flatc" \
                      "$ET_ROOT/build/third-party/flatc_ep/bin/flatc"; do
  if [ -x "$flatc_candidate" ]; then FLATC="$flatc_candidate"; break; fi
done
if [ -n "$FLATC" ]; then
  "$FLATC" --cpp --cpp-std c++11 --gen-mutable --scoped-enums \
    -o "$ET_SRC/schema/" \
    "$ET_ROOT/schema/program.fbs" \
    "$ET_ROOT/schema/scalar_type.fbs" 2>/dev/null
  echo "  Schema headers regenerated"
fi

echo "[7/7] Arduino patches applied"

# ─────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────
NSRC=$(find "$OUT_DIR/src" -name "*.cpp" -o -name "*.c" | wc -l | tr -d ' ')
NHDR=$(find "$OUT_DIR/src" -name "*.h" | wc -l | tr -d ' ')

echo ""
echo "=== Library built ==="
echo "  Location:  $OUT_DIR"
echo "  Sources:   $NSRC"
echo "  Headers:   $NHDR"
echo ""
echo "Install:"
echo "  cp -r $OUT_DIR ~/Arduino/libraries/"
echo ""
echo "Or clean up:"
echo "  $0 --clean"
