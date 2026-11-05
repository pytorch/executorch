#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# build_idf_component.sh -- Generate the ExecuTorch ESP-IDF component
# (source-vendored, registry-shaped) from this repository.
#
# Usage:
#   ./build_idf_component.sh --out /path/to/components/executorch
#   ./build_idf_component.sh --version
#   ./build_idf_component.sh --out /path/to/components/executorch --clean
#
# Derived from examples/arduino/build_arduino_library.sh with an ESP-IDF
# output layout and the shared ESP-IDF PAL. The component vendors every
# kernel source of its kernel libraries (portable, quantized and the
# Cadence backend's generic kernels) but registers none of them: a
# project's kernel bundle, made by the component's tools/gen_kernels.py
# from its models, selects the kernels at build time.
#
# Host requirements: python3 (to read build_variables.bzl), cmake (to
# build flatc from third-party/flatbuffers when no flatc is available),
# and the git submodule third-party/flatbuffers initialized. Consumers of
# the output need only ESP-IDF, plus an ExecuTorch Python environment to
# make kernel bundles. Set PYTHON=... / FLATC=... to override discovery.
#
# NOTE: This script is coupled to the ExecuTorch source tree layout. Its
# runtime source lists come from the same build_variables.bzl the CMake
# build uses, and kernel sources are globbed, so upstream file moves are
# picked up automatically. Consumer builds for Xtensa + RISC-V targets
# are the safety net.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ET_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEMPLATE_DIR="$SCRIPT_DIR/idf_component_template"
ESP_PAL_SRC="$SCRIPT_DIR/platform/esp_pal.cpp"
PYTHON="${PYTHON:-python3}"
PACKAGING_REV=3 # bump when the packaging (not ExecuTorch) changes

OUT_DIR=""
CLEAN=false

usage() {
  echo "Usage: $0 --out <dir> [--clean]" >&2
  echo "       $0 --version" >&2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT_DIR="$2"; shift 2 ;;
    --clean) CLEAN=true; shift ;;
    --version)
      echo "$(cat "$ET_ROOT/version.txt")~$PACKAGING_REV"
      exit 0
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [ -z "$OUT_DIR" ]; then
  echo "ERROR: --out is required." >&2; usage; exit 1
fi
OUT_DIR_AS_GIVEN="$OUT_DIR"
# Normalize to an absolute path (dir may not exist yet).
case "$OUT_DIR" in
  /*) ;;
  *) OUT_DIR="$PWD/$OUT_DIR" ;;
esac

if [ "$CLEAN" = true ]; then
  if [ -e "$OUT_DIR/et_generated.cmake" ] || [ -e "$OUT_DIR/.et-generated-component" ]; then
    rm -rf "$OUT_DIR"
    echo "Cleaned $OUT_DIR"
  else
    echo "Nothing to clean: $OUT_DIR is not a generated component." >&2
  fi
  exit 0
fi

ET_VERSION=$(cat "$ET_ROOT/version.txt")
if [[ ! "$ET_VERSION" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]; then
  echo "ERROR: version.txt must contain an ExecuTorch version in X.Y.Z form." >&2
  exit 1
fi
ET_COMPONENT_VERSION="${ET_VERSION}~${PACKAGING_REV}"

if [ ! -f "$ET_ROOT/third-party/flatbuffers/LICENSE" ]; then
  echo "ERROR: third-party/flatbuffers is empty. Run: git submodule update --init third-party/flatbuffers" >&2
  exit 1
fi

# Refuse to clobber a directory that is not a previous run's output.
if [ -d "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ] \
   && [ ! -e "$OUT_DIR/et_generated.cmake" ] \
   && [ ! -e "$OUT_DIR/.et-generated-component" ]; then
  echo "ERROR: $OUT_DIR exists and is not a generated component; refusing to overwrite." >&2
  exit 1
fi

echo "=== Building ExecuTorch ESP-IDF component ==="
echo "  ET repo:   $ET_ROOT"
echo "  Version:   $ET_COMPONENT_VERSION (ExecuTorch $ET_VERSION)"
echo "  Output:    $OUT_DIR"

# Print one source list from build/build_variables.bzl (same mechanism
# as tools/cmake/Codegen.cmake:executorch_append_filelist).
bzl_list() {
  "$PYTHON" -c "
exec(open('$ET_ROOT/shim_et/xplat/executorch/build/build_variables.bzl').read())
for f in $1: print(f)
"
}

SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

# ─────────────────────────────────────────────────────────
# 1. Generate schema headers with flatc (host tool only;
#    consumers never need it). Same flags + inputs as
#    schema/CMakeLists.txt and extension/flat_tensor.
# ─────────────────────────────────────────────────────────
FLATC="${FLATC:-}"
if [ -z "$FLATC" ]; then
  if command -v flatc >/dev/null 2>&1; then
    FLATC="$(command -v flatc)"
  else
    for cand in "$ET_ROOT/cmake-out/third-party/flatc_ep/bin/flatc" \
                "$ET_ROOT/cmake-out-mac/third-party/flatc_ep/bin/flatc" \
                "$ET_ROOT/build/third-party/flatc_ep/bin/flatc"; do
      if [ -x "$cand" ]; then FLATC="$cand"; break; fi
    done
  fi
fi
if [ -z "$FLATC" ]; then
  echo "  Building flatc from third-party/flatbuffers ..."
  cmake -S "$ET_ROOT/third-party/flatbuffers" -B "$SCRATCH/flatc-build" \
    -DCMAKE_BUILD_TYPE=Release -DFLATBUFFERS_BUILD_TESTS=OFF \
    -DFLATBUFFERS_BUILD_FLATLIB=OFF -DFLATBUFFERS_BUILD_SHAREDLIB=OFF \
    -DFLATBUFFERS_INSTALL=OFF > "$SCRATCH/flatc-config.log" 2>&1
  cmake --build "$SCRATCH/flatc-build" --target flatc -j 4 \
    > "$SCRATCH/flatc-build.log" 2>&1 || {
      echo "ERROR: flatc build failed; see $SCRATCH/flatc-build.log" >&2
      trap - EXIT; exit 1; }
  FLATC="$SCRATCH/flatc-build/flatc"
fi
echo "  flatc: $FLATC"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
# Marker so a partial output from a failed run is recognised (and may
# be overwritten) by the overwrite guard above.
echo "generated by build_idf_component.sh (in progress)" \
  > "$OUT_DIR/.et-generated-component"
mkdir -p "$OUT_DIR/include/executorch/schema" \
         "$OUT_DIR/include/executorch/extension/flat_tensor/serialize" \
         "$OUT_DIR/src/executorch"

( cd "$ET_ROOT/schema" && "$FLATC" --cpp --cpp-std c++11 --gen-mutable \
    --scoped-enums -o "$OUT_DIR/include/executorch/schema" scalar_type.fbs )
( cd "$ET_ROOT/schema" && "$FLATC" --cpp --cpp-std c++11 --gen-mutable \
    --scoped-enums -o "$OUT_DIR/include/executorch/schema" program.fbs )
( cd "$ET_ROOT/extension/flat_tensor/serialize" && "$FLATC" --cpp \
    --cpp-std c++11 --gen-mutable --scoped-enums \
    -o "$OUT_DIR/include/executorch/extension/flat_tensor/serialize" \
    scalar_type.fbs flat_tensor.fbs )

for h in "$OUT_DIR/include/executorch/schema/program_generated.h" \
         "$OUT_DIR/include/executorch/extension/flat_tensor/serialize/flat_tensor_generated.h"; do
  if [ ! -f "$h" ]; then
    echo "ERROR: flatc did not produce $h" >&2; exit 1
  fi
done
# program_generated.h must include its sibling scalar_type header in a
# way that resolves through <executorch/schema/...> or same-directory;
# anything else means the include layout drifted and builds will break.
if grep -E '#include ".*scalar_type_generated\.h"' \
     "$OUT_DIR/include/executorch/schema/program_generated.h" \
     | grep -qvE '#include "(executorch/schema/)?scalar_type_generated\.h"'; then
  echo "ERROR: unexpected include form in program_generated.h" >&2; exit 1
fi
echo "[1/5] Schema headers generated"

# ─────────────────────────────────────────────────────────
# 2. Vendor sources. Runtime + extension sources come from
#    build_variables.bzl lists; kernel sources are globbed.
#    Output mirrors upstream paths under src/executorch/.
# ─────────────────────────────────────────────────────────
copy_srcs() { # ET_ROOT-relative paths on stdin; $1 = list file to append to
  local f
  while IFS= read -r f; do
    if [ ! -f "$ET_ROOT/$f" ]; then
      echo "ERROR: missing source file: $f" >&2; exit 1
    fi
    mkdir -p "$OUT_DIR/src/executorch/$(dirname "$f")"
    cp "$ET_ROOT/$f" "$OUT_DIR/src/executorch/$f"
    echo "src/executorch/$f" >> "$1"
  done
}

: > "$SCRATCH/core_srcs.txt"
: > "$SCRATCH/kernel_srcs.txt"
bzl_list EXECUTORCH_CORE_SRCS | copy_srcs "$SCRATCH/core_srcs.txt"
# register_prim_ops.cpp is compiled by the build, with the bundle's prim
# selection, only when a project has a kernel bundle.
bzl_list EXECUTORCH_SRCS | grep -v '/register_prim_ops\.cpp$' \
  | copy_srcs "$SCRATCH/kernel_srcs.txt"
echo "kernels/prim_ops/register_prim_ops.cpp" | copy_srcs "$SCRATCH/prim_src.txt"
( cd "$ET_ROOT" && find kernels/portable/cpu kernels/quantized/cpu \
    backends/cadence/generic -name '*.cpp' -not -path '*/test*/*' | sort ) \
  | copy_srcs "$SCRATCH/kernel_srcs.txt"

echo "[2/5] Runtime, extension and kernel sources copied"

# ─────────────────────────────────────────────────────────
# 3. Vendor headers: executorch/** and c10/torch (public),
#    flatbuffers (private: only internal headers include it).
# ─────────────────────────────────────────────────────────
copy_headers() { # $1=src dir under ET_ROOT, $2=dest under OUT/include
  local src_dir="$1" dest_dir="$2" f
  ( cd "$ET_ROOT" && find "$src_dir" \( -name '*.h' -o -name '*.hpp' \) \
      -not -path '*/test*/*' ) | while IFS= read -r f; do
    local rel="${f#"$src_dir"/}"
    mkdir -p "$OUT_DIR/include/$dest_dir/$(dirname "$rel")"
    cp "$ET_ROOT/$f" "$OUT_DIR/include/$dest_dir/$rel"
  done
}

copy_headers runtime executorch/runtime
# c10/torch live inside runtime/core/portable_type/c10 upstream but are
# included as <c10/...> and <torch/...>; the copies above just placed
# them under executorch/runtime/... where nothing looks. Move them to
# the include root instead.
rm -rf "$OUT_DIR/include/executorch/runtime/core/portable_type/c10"
cp -r "$ET_ROOT/runtime/core/portable_type/c10/c10" "$OUT_DIR/include/c10"
cp -r "$ET_ROOT/runtime/core/portable_type/c10/torch" "$OUT_DIR/include/torch"
find "$OUT_DIR/include/c10" "$OUT_DIR/include/torch" \
  \( -name '*.cpp' -o -name 'BUILD*' -o -name '*.bzl' \) -delete 2>/dev/null || true

# cmake_macros.h is generated by the PyTorch build; stub it like the
# Arduino packaging does (C10_USING_CUSTOM_GENERATED_MACROS is defined
# publicly, this covers headers that include it unconditionally).
mkdir -p "$OUT_DIR/include/torch/headeronly/macros"
cat > "$OUT_DIR/include/torch/headeronly/macros/cmake_macros.h" <<'STUB'
#pragma once
#define C10_BUILD_SHARED_LIBS
#define C10_USE_GLOG 0
#define C10_USE_MINIMAL_GLOG 0
#define C10_USE_GFLAGS 0
STUB

copy_headers schema executorch/schema
# Only the header-only BufferDataLoader; Module, the file/mmap loaders
# and .ptd data maps are not vendored.
mkdir -p "$OUT_DIR/include/executorch/extension/data_loader"
cp "$ET_ROOT/extension/data_loader/buffer_data_loader.h" \
   "$OUT_DIR/include/executorch/extension/data_loader/"
copy_headers kernels/portable executorch/kernels/portable
copy_headers kernels/quantized executorch/kernels/quantized
copy_headers kernels/prim_ops executorch/kernels/prim_ops
copy_headers backends/cadence/generic executorch/backends/cadence/generic
# register_prim_ops.cpp includes selective_build_prim_ops.h by its bare
# name, which resolves next to the source.
cp "$ET_ROOT/kernels/prim_ops/selective_build_prim_ops.h" \
   "$OUT_DIR/src/executorch/kernels/prim_ops/"

mkdir -p "$OUT_DIR/priv_include"
cp -r "$ET_ROOT/third-party/flatbuffers/include/flatbuffers" \
      "$OUT_DIR/priv_include/flatbuffers"

echo "[3/5] Headers and third-party code copied"

# ─────────────────────────────────────────────────────────
# 4. Kernel bundle tooling: gen_kernels.py, the kernel
#    libraries' yamls and ExecuTorch's codegen templates.
#    Library order is gen_kernels.py's assignment priority.
# ─────────────────────────────────────────────────────────
LIBRARIES=(
  "portable kernels/portable/functions.yaml"
  "quantized kernels/quantized/quantized.yaml"
  "cadence backends/cadence/aot/functions.yaml"
)
mkdir -p "$OUT_DIR/tools/libraries" "$OUT_DIR/tools/codegen"
cp "$TEMPLATE_DIR/tools/gen_kernels.py" "$OUT_DIR/tools/"
cp -r "$ET_ROOT/codegen/templates" "$OUT_DIR/tools/codegen/templates"
{
  echo '{'
  echo '  "schema": 1,'
  echo '  "prim_ops_source": "src/executorch/kernels/prim_ops/register_prim_ops.cpp",'
  echo '  "libraries": ['
  sep=""
  for entry in "${LIBRARIES[@]}"; do
    read -r name yaml <<< "$entry"
    cp "$ET_ROOT/$yaml" "$OUT_DIR/tools/libraries/$name.yaml"
    printf '%s    {"name": "%s", "yaml": "tools/libraries/%s.yaml"}' "$sep" "$name" "$name"
    sep=$',\n'
  done
  echo ''
  echo '  ]'
  echo '}'
} > "$OUT_DIR/tools/kernel_libraries.json"

echo "[4/5] Kernel bundle tooling copied"

# ─────────────────────────────────────────────────────────
# 5. Component glue from idf_component_template/ and the shared
#    ESP-IDF PAL, plus generated metadata (et_generated.cmake,
#    Kconfig, idf_component.yml, VERSION, licenses).
# ─────────────────────────────────────────────────────────
cp "$TEMPLATE_DIR/CMakeLists.txt" "$TEMPLATE_DIR/project_include.cmake" \
   "$TEMPLATE_DIR/README.md" "$OUT_DIR/"
cp "$TEMPLATE_DIR/include/et_model.h" "$OUT_DIR/include/"
mkdir -p "$OUT_DIR/src/platform"
cp "$TEMPLATE_DIR/src/et_model.cpp" "$OUT_DIR/src/"
cp "$ESP_PAL_SRC" "$OUT_DIR/src/platform/"
cp "$ET_ROOT/LICENSE" "$OUT_DIR/LICENSE"

LICENSES="$OUT_DIR/THIRD_PARTY_LICENSES"
mkdir -p "$LICENSES"
cp "$ET_ROOT/third-party/flatbuffers/LICENSE" "$LICENSES/flatbuffers-LICENSE.txt"
cat > "$LICENSES/README.md" <<'NOTICE'
# Third-party licenses

This component redistributes source from the projects below.
ExecuTorch's own BSD license is in the LICENSE file at the root.

| Component | Location in this component | License |
|---|---|---|
| FlatBuffers (Google) | `priv_include/flatbuffers/` | Apache-2.0 — `flatbuffers-LICENSE.txt` |
| PyTorch c10 (Meta) | `include/c10/`, `include/torch/` | BSD-3-Clause, as exact copies from PyTorch core |

Kernel registration that `tools/gen_kernels.py` generates from PyTorch's
`native_functions.yaml` and the templates in `tools/codegen/templates/`
carries the same terms as ExecuTorch itself.
NOTICE

# et_generated.cmake: baked values consumed by CMakeLists.txt.
{
  echo "# GENERATED by examples/espressif/build_idf_component.sh -- do not edit."
  echo "set(ET_COMPONENT_VERSION \"$ET_COMPONENT_VERSION\")"
  echo "set(ET_CORE_SRCS"
  sort -u "$SCRATCH/core_srcs.txt" | sed 's/.*/    "&"/'
  echo ")"
  echo "set(ET_KERNEL_SRCS"
  sort -u "$SCRATCH/kernel_srcs.txt" | sed 's/.*/    "&"/'
  echo ")"
  echo "set(ET_PRIM_OPS_SRC \"$(cat "$SCRATCH/prim_src.txt")\")"
} > "$OUT_DIR/et_generated.cmake"

cat > "$OUT_DIR/Kconfig" <<'KCONFIG'
# GENERATED by examples/espressif/build_idf_component.sh -- do not edit.
menu "ExecuTorch"

    config ET_ENABLE_LOGGING
        bool "Enable ExecuTorch runtime logging"
        default y
        help
            When disabled, ET_LOG is compiled out entirely
            (ET_LOG_ENABLED=0). When enabled, messages go through
            ESP_LOG under the "executorch" tag. Newlib's nano printf
            (LIBC_NEWLIB_NANO_FORMAT) cannot format the 64-bit integers
            in ExecuTorch's error messages, so with it logging is
            compiled out regardless, with a build warning.

    choice ET_LOG_LEVEL
        prompt "ExecuTorch log level floor"
        default ET_LOG_LEVEL_INFO
        depends on ET_ENABLE_LOGGING
        help
            Compile-time floor (ET_MIN_LOG_LEVEL), also applied in
            the PAL, on top of the usual IDF runtime log levels.

        config ET_LOG_LEVEL_ERROR
            bool "Error"
        config ET_LOG_LEVEL_WARN
            bool "Warning"
        config ET_LOG_LEVEL_INFO
            bool "Info"
        config ET_LOG_LEVEL_DEBUG
            bool "Debug"
    endchoice

    config ET_OPTIMIZE_SIZE
        bool "Build the runtime with -Os"
        default y
        help
            Optimize the ExecuTorch runtime and kernels for size
            regardless of the project's compiler optimization level.
            This trades speed for flash; turn it off when inference
            time matters more.

    config ET_PROGRAM_VERIFICATION
        bool "Compile in full .pte verification"
        default n
        help
            Required for et_model_config_t.verify; without it
            et_model_create rejects verify requests.

    config ET_DEBUG_ASSERTS
        bool "Keep ExecuTorch debug assertions"
        default n
        help
            When disabled, the runtime builds with NDEBUG.

    config ET_MAX_KERNEL_NUM
        int "Kernel registry size (0 = the kernel bundle's kernel count)"
        default 0
        help
            Raise this when the application registers its own kernels.
            Must not be below the kernel bundle's kernel count.

endmenu
KCONFIG

sed "s/@ET_COMPONENT_VERSION@/$ET_COMPONENT_VERSION/" \
  "$TEMPLATE_DIR/idf_component.yml.in" > "$OUT_DIR/idf_component.yml"

ET_SHA=$(git -C "$ET_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")
ET_DIRTY=$(git -C "$ET_ROOT" status --porcelain 2>/dev/null | head -1)
{
  echo "# GENERATED by examples/espressif/build_idf_component.sh -- do not edit."
  echo "executorch_version: $ET_VERSION"
  echo "component_version: $ET_COMPONENT_VERSION"
  echo "executorch_commit: $ET_SHA${ET_DIRTY:+ (tree had uncommitted changes)}"
  echo "generator: examples/espressif/build_idf_component.sh --out $OUT_DIR_AS_GIVEN"
  echo "kernel_libraries:"
  for entry in "${LIBRARIES[@]}"; do
    read -r name yaml <<< "$entry"
    sha=$("$PYTHON" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" \
      "$OUT_DIR/tools/libraries/$name.yaml")
    echo "  $name: {yaml: $yaml, sha256: $sha}"
  done
  echo "substitutions:"
  echo "  - torch/headeronly/macros/cmake_macros.h: packaging stub (as in examples/arduino)"
  echo "notes:"
  echo "  - kernels are registered from the project's kernel bundle (tools/gen_kernels.py)"
  echo "  - cadence:: ops use the Cadence backend's generic (portable) kernels (backends/cadence/generic), not the HiFi/xt-clang kernels"
  echo "  - no .pte models are included; models ship with the application"
} > "$OUT_DIR/VERSION"

echo "[5/5] Component glue and metadata written"

# Sanity checks.
for f in "CMakeLists.txt" "project_include.cmake" "et_generated.cmake" \
         "Kconfig" "idf_component.yml" "VERSION" "LICENSE" "include/et_model.h" \
         "include/executorch/runtime/executor/program.h" \
         "include/executorch/schema/program_generated.h" \
         "src/et_model.cpp" "src/platform/esp_pal.cpp" \
         "tools/gen_kernels.py" "tools/kernel_libraries.json"; do
  if [ ! -f "$OUT_DIR/$f" ]; then
    echo "ERROR: generated component is missing $f" >&2; exit 1
  fi
done
# Every source named in et_generated.cmake must exist on disk.
grep -o '"src/[^"]*"' "$OUT_DIR/et_generated.cmake" | tr -d '"' \
  | while IFS= read -r f; do
      if [ ! -f "$OUT_DIR/$f" ]; then
        echo "ERROR: et_generated.cmake names missing file: $f" >&2; exit 1
      fi
    done

# Success: et_generated.cmake is the marker now.
rm -f "$OUT_DIR/.et-generated-component"

NSRC=$(find "$OUT_DIR/src" \( -name '*.cpp' -o -name '*.c' \) | wc -l | tr -d ' ')
NHDR=$(find "$OUT_DIR/include" \( -name '*.h' -o -name '*.hpp' \) | wc -l | tr -d ' ')
SIZE=$(du -sh "$OUT_DIR" | awk '{print $1}')

echo ""
echo "=== ExecuTorch ESP-IDF component $ET_COMPONENT_VERSION ==="
echo "  Output:    $OUT_DIR ($SIZE)"
echo "  Sources:   $NSRC, headers: $NHDR"
echo ""
echo "Next: make a kernel bundle for your models with"
echo "  python $OUT_DIR_AS_GIVEN/tools/gen_kernels.py --pte model.pte --out main/et_kernels"
echo "and declare both in your project; see $OUT_DIR_AS_GIVEN/README.md."
