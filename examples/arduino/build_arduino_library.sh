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
# Output: arduino_lib/ExecuTorch/ (self-contained, installable)
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
OUT_DIR="$SCRIPT_DIR/arduino_lib/ExecuTorch"
PROPS="$SCRIPT_DIR/library.properties"
PYTHON="${PYTHON:-python3}"

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
cp "$SCRIPT_DIR/ExecuTorch.h" "$OUT_DIR/src/"
cp "$SCRIPT_DIR/platform_stubs.c" "$OUT_DIR/src/"
cp -r "$SCRIPT_DIR/examples/"* "$OUT_DIR/examples/"
# Training checkpoints are how a model is regenerated, not something the
# library needs at runtime.
find "$OUT_DIR/examples" -name "*.pth" -delete

# Tooling the README points users at. None of it is reachable once the library
# is installed on its own, so it travels with the library under extras/, which
# the Arduino spec excludes from the build.
mkdir -p "$OUT_DIR/extras/tools"
cp "$SCRIPT_DIR/build_arduino_library.sh" \
   "$SCRIPT_DIR/pte_to_header.py" \
   "$SCRIPT_DIR/export_model.py" \
   "$SCRIPT_DIR/generate_test_input.py" "$OUT_DIR/extras/tools/"

# An example that ships a model.pte gets a model.h generated from it. Without
# one the sketch #errors the moment it is opened from the IDE menu. Generated
# rather than checked in so the header cannot drift from its .pte.
for pte in "$OUT_DIR/examples/"*/model.pte; do
  [ -e "$pte" ] || continue
  "$PYTHON" "$SCRIPT_DIR/pte_to_header.py" \
    --pte "$pte" --output "$(dirname "$pte")/model.h"
  rm -f "$pte"
done

echo "[1/7] Metadata, examples and tooling copied"

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
cp "$ET_ROOT/schema/"*.cpp "$ET_SRC/schema/" 2>/dev/null || true
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
# 3b. Generate the kernel registration translation unit.
#
# Kernels register through codegen, not through self-registering static
# initializers. Without this the ops compile but never reach the operator
# registry, and every Method::load fails with OperatorMissing.
# ─────────────────────────────────────────────────────────
#
# The op set is a size decision, not a detail. Registering every portable op
# costs 1.58 MB of text — twice the Uno Q's 786 KB flash. The default below is
# the Cortex-M op set plus the portable ops a quantized CNN still needs.
# Override with ROOT_OPS="aten::foo.out,..." or ALL_OPS=1 when the target has
# room to spare.
if ! TORCHGEN=$("$PYTHON" -c "import torchgen, os; print(os.path.dirname(torchgen.__file__))" 2>/dev/null); then
  echo "ERROR: cannot import torchgen with $PYTHON."
  echo "       The kernel registration codegen needs it. Run ./install_executorch.sh,"
  echo "       or set PYTHON=... to an interpreter that has ExecuTorch installed."
  exit 1
fi
for yaml in "$TORCHGEN/packaged/ATen/native/tags.yaml" \
            "$TORCHGEN/packaged/ATen/native/native_functions.yaml"; do
  if [ ! -f "$yaml" ]; then
    echo "ERROR: $yaml is missing from the torchgen at $TORCHGEN."
    echo "       That install looks incomplete; reinstall with ./install_executorch.sh."
    exit 1
  fi
done

# The exporter and the runtime must be the same ExecuTorch. A pip release wheel
# can be months behind this checkout, and a model exported against one schema
# fails at Method::execute against a library built from another.
ET_PY=$("$PYTHON" -c "import executorch; print(next(iter(executorch.__path__), ''))" 2>/dev/null || echo "")
case "$ET_PY" in
  "$ET_ROOT"*) ;;
  "") echo "  NOTE: no executorch Python package found; the library will build but" ;
      echo "        you cannot export models with this interpreter." ;;
  *)  echo "  WARNING: $PYTHON imports executorch from $ET_PY, not $ET_ROOT." ;
      echo "           Models exported with it may not match this library. See" ;
      echo "           'Keeping this working' in examples/arduino/README.md." ;;
esac

CODEGEN_OUT="$ET_SRC/codegen"
CORTEX_M_YAML="$ET_ROOT/backends/cortex_m/ops/operators.yaml"
mkdir -p "$CODEGEN_OUT"

# dim_order_ops are not optional: the Cortex-M lowering emits
# _clone_dim_order in place of aten::clone for channels-last models.
DEFAULT_ROOT_OPS="aten::add.out,aten::mul.out,aten::sub.out,aten::div.out,\
aten::view_copy.out,aten::permute_copy.out,aten::clone.out,aten::cat.out,\
aten::slice_copy.Tensor_out,aten::_softmax.out,aten::mean.out,aten::relu.out,\
dim_order_ops::_clone_dim_order.out,dim_order_ops::_to_dim_order_copy.out"
ROOT_OPS="${ROOT_OPS:-$DEFAULT_ROOT_OPS}"

if [ "${ALL_OPS:-0}" = "1" ]; then
  OPLIST_SELECTION=(--include_all_operators)
  echo "  Op set: every portable op (large - verify it fits your target)"
else
  OPLIST_SELECTION=(--root_ops="$ROOT_OPS")
  echo "  Op set: default ($(echo "$ROOT_OPS" | tr ',' '\n' | wc -l | tr -d ' ') root ops)"
fi

( cd "$ET_ROOT" && \
  "$PYTHON" -m codegen.tools.gen_oplist \
    --output_path="$CODEGEN_OUT/selected_operators.yaml" \
    --ops_schema_yaml_path="$CORTEX_M_YAML" \
    "${OPLIST_SELECTION[@]}" && \
  "$PYTHON" -m codegen.gen \
    --source-path="$ET_ROOT/codegen" \
    --install-dir="$CODEGEN_OUT" \
    --tags-path="$TORCHGEN/packaged/ATen/native/tags.yaml" \
    --aten-yaml-path="$TORCHGEN/packaged/ATen/native/native_functions.yaml" \
    --op-selection-yaml-path="$CODEGEN_OUT/selected_operators.yaml" \
    --functions-yaml-path="$ET_ROOT/kernels/portable/functions.yaml" \
    --custom-ops-yaml-path="$CORTEX_M_YAML" ) > /dev/null

# Right-size the operator registry. Without this header it falls back to a
# fixed MAX_KERNEL_NUM sized for a much larger build, which costs RAM the board
# does not have to spare.
( cd "$ET_ROOT" && "$PYTHON" -m codegen.tools.gen_max_kernel_num \
  --oplist-yaml="$CODEGEN_OUT/selected_operators.yaml" \
  --prim-ops-source="$ET_ROOT/kernels/prim_ops/register_prim_ops.cpp" \
  --output-path="$ET_SRC/runtime/kernel/selected_max_kernel_num.h" )

# gen writes the same content to both names; keeping both is a duplicate-symbol error.
rm -f "$CODEGEN_OUT/RegisterCodegenUnboxedKernels_0.cpp"
rm -f "$CODEGEN_OUT/selected_operators.yaml"
# These register custom ops into PyTorch, not into the ET runtime. They pull in
# <torch/library.h> and <ATen/Tensor.h>, which do not exist on device.
rm -f "$CODEGEN_OUT/RegisterCPUCustomOps.cpp" \
      "$CODEGEN_OUT/RegisterCPUStub.cpp" \
      "$CODEGEN_OUT/RegisterSchema.cpp" \
      "$CODEGEN_OUT/CustomOpsNativeFunctions.h"

if [ ! -f "$CODEGEN_OUT/RegisterCodegenUnboxedKernelsEverything.cpp" ]; then
  echo "ERROR: kernel registration codegen produced no output."
  exit 1
fi

echo "[3b/7] Kernel registration generated"

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
  # Bindings are pybind11 host code and cannot be cross-compiled.
  rm -rf "$OUT_DIR/src/cmsis-nn/Source/Bindings"
  find "$OUT_DIR/src/cmsis-nn" -name "CMakeLists.txt" -delete
  # Arduino compiles every source under src/ with no way to pass per-library
  # defines, so drop the float extensions that ARM_NN_ENABLE_F32/F16 gate off
  # by default. They need CMSIS-DSP types the Cortex-M backend never uses.
  find "$OUT_DIR/src/cmsis-nn/Source" \
    \( -name "*_f16.c" -o -name "*_f32.c" -o -name "*_flt.c" \) -delete
  cp "$CMSIS_NN/LICENSE" "$OUT_DIR/src/cmsis-nn/"
  cp "$CMSIS_NN/Include/"*.h "$OUT_DIR/src/" 2>/dev/null || true
  if [ -d "$CMSIS_NN/Include/Internal" ]; then
    mkdir -p "$OUT_DIR/src/Internal"
    cp "$CMSIS_NN/Include/Internal/"*.h "$OUT_DIR/src/Internal/"
  fi
  CMSIS_NN_REV=$(git -C "$CMSIS_NN" rev-parse HEAD 2>/dev/null || echo "unknown")
  echo "[5/7] CMSIS-NN copied from $CMSIS_NN"
else
  CMSIS_NN_REV="absent"
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

# ─────────────────────────────────────────────────────────
# Third-party notices. The vendored trees are redistributed in source form,
# so Apache-2.0 section 4 and the PyTorch BSD terms require their licenses
# to travel with them.
# ─────────────────────────────────────────────────────────
LICENSES="$OUT_DIR/extras/THIRD_PARTY_LICENSES"
mkdir -p "$LICENSES"
for dep in flatbuffers flatcc; do
  if [ ! -f "$ET_ROOT/third-party/$dep/LICENSE" ]; then
    echo "ERROR: third-party/$dep is empty. Run: git submodule update --init third-party/$dep"
    exit 1
  fi
  cp "$ET_ROOT/third-party/$dep/LICENSE" "$LICENSES/$dep-LICENSE.txt"
done
if [ -n "$CMSIS_NN" ]; then
  cp "$CMSIS_NN/LICENSE" "$LICENSES/CMSIS-NN-LICENSE.txt"
fi

cat > "$LICENSES/README.md" << 'NOTICE'
# Third-party licenses

This library redistributes source from the projects below. ExecuTorch's own
BSD license is in the LICENSE file at the root.

| Component | Location in this library | License |
|---|---|---|
| CMSIS-NN (Arm) | `src/cmsis-nn/`, `src/arm_nn*.h`, `src/Internal/` | Apache-2.0 — `CMSIS-NN-LICENSE.txt` |
| FlatBuffers (Google) | `src/flatbuffers/` | Apache-2.0 — `flatbuffers-LICENSE.txt` |
| flatcc (Mikkel F. Jorgensen) | `src/flatcc/` | Apache-2.0 — `flatcc-LICENSE.txt` |
| PyTorch c10 (Meta) | `src/c10/`, `src/torch/` | BSD-3-Clause, as exact copies from PyTorch core |

`src/executorch/codegen/` is generated by ExecuTorch's codegen from PyTorch's
`native_functions.yaml` and carries the same terms as ExecuTorch itself.
NOTICE

echo "[6/7] Third-party dependencies copied"

# ─────────────────────────────────────────────────────────
# 6. Apply Arduino-specific patches
# ─────────────────────────────────────────────────────────

# Fix: #include <exception> before <variant> in all ET headers
find "$OUT_DIR/src/executorch" -name "*.h" -print0 | \
  xargs -0 perl -pi -e 's/#include <variant>/#include <exception>\n#include <variant>/g'

# Remove test files, ATen-specific files, non-Zephyr platform backends
find "$OUT_DIR" -path "*testing*" -delete 2>/dev/null || true
# ATen-mode sources only. *_exec_aten.cpp is portable-mode and required.
find "$OUT_DIR" -name "*_aten.cpp" ! -name "*_exec_aten.cpp" -delete 2>/dev/null || true
find "$OUT_DIR" -path "*test*" -name "*.cpp" -delete 2>/dev/null || true
rm -f "$OUT_DIR/src/executorch/runtime/platform/default/android.cpp"
rm -f "$OUT_DIR/src/executorch/runtime/platform/default/posix.cpp"
rm -f "$OUT_DIR/src/executorch/runtime/platform/default/windows.cpp"
# minimal.cpp and zephyr.cpp both define the et_pal_* backend, so shipping both
# leaves the choice to link order. minimal's logger is an empty body and its
# et_pal_allocate returns nullptr, which silently discards every ET_LOG.
rm -f "$OUT_DIR/src/executorch/runtime/platform/default/minimal.cpp"

# zephyr.cpp logs through fprintf, and platform_stubs.c stubs fprintf out to
# nothing, so runtime diagnostics never reach the user. Route them to a weak
# hook a sketch can implement -- see the examples for a Serial implementation.
ZEPHYR_PAL="$OUT_DIR/src/executorch/runtime/platform/default/zephyr.cpp"
"$PYTHON" - "$ZEPHYR_PAL" << 'PATCH'
import sys
p = sys.argv[1]
s = open(p).read()
old = """  fprintf(
      stderr,
      "%c [executorch:%s:%zu %s()] %s\\n",
      level,
      filename,
      line,
      function,
      message);"""
new = """  char et_log_buf[256];
  snprintf(
      et_log_buf,
      sizeof(et_log_buf),
      "%c [ET:%s:%zu] %s",
      (char)level,
      filename,
      line,
      message);
  et_arduino_log(et_log_buf);"""
if old not in s:
    sys.exit("ERROR: zephyr.cpp log call not found; the PAL changed upstream.")
s = s.replace(old, new)
s = s.replace(
    "void et_pal_emit_log_message(",
    'extern "C" __attribute__((weak)) void et_arduino_log(const char*) {}\n\n'
    "void et_pal_emit_log_message(",
    1,
)
open(p, "w").write(s)
PATCH

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
# Record what produced this tree. The published library is a generated
# artifact, so without this there is no way back to the sources.
# ─────────────────────────────────────────────────────────
ET_SHA=$(git -C "$ET_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")
ET_DIRTY=$(git -C "$ET_ROOT" status --porcelain "$SCRIPT_DIR" 2>/dev/null | head -1)
cat > "$OUT_DIR/extras/PROVENANCE.txt" << PROV
This library is generated, not hand-written. Everything under src/ was copied
out of ExecuTorch by examples/arduino/build_arduino_library.sh, and the
model.h in each example was converted from a .pte exported by that same
checkout. Do not edit either by hand; regenerate instead.

executorch:  https://github.com/pytorch/executorch
commit:      $ET_SHA${ET_DIRTY:+ (tree had uncommitted changes under examples/arduino)}
CMSIS-NN:    $CMSIS_NN_REV
op set:      $([ "${ALL_OPS:-0}" = "1" ] && echo "all portable ops" || echo "$ROOT_OPS")
kernels:     $(grep -c 'Kernel(' "$CODEGEN_OUT/RegisterCodegenUnboxedKernelsEverything.cpp")

The commit above is not decoration. Cortex-M operator schemas change between
ExecuTorch releases, and a model exported against one commit fails at
Method::execute against a library built from another - it loads fine, resolves
every operator, then returns InvalidProgram (0x23). The library and the models
it ships must come from this one commit.

To regenerate:

  git -C <executorch> checkout \$(cat executorch_pin.txt)
  ./install_executorch.sh                     # so the exporter matches too
  examples/arduino/build_arduino_library.sh

To move to a newer ExecuTorch, bump executorch_pin.txt, regenerate, and
re-export the example models in the same change. Override the op set with
ROOT_OPS="aten::foo.out,..." or ALL_OPS=1.
PROV

# The pin is the input a maintainer edits; PROVENANCE records what was used.
# One SHA per file, matching .ci/docker/ci_commit_pins/ in ExecuTorch.
echo "$ET_SHA" > "$OUT_DIR/executorch_pin.txt"

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
