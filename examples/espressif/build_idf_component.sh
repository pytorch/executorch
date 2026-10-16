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
#   ./build_idf_component.sh --profiles wakeword_int8 --out /path/to/components/executorch
#   ./build_idf_component.sh --version
#   ./build_idf_component.sh --out /path/to/components/executorch --clean
#
# Derived from examples/arduino/build_arduino_library.sh with an ESP-IDF
# output layout, the shared ESP-IDF PAL, and per-profile kernel
# registration: each
# profile in idf_profiles/<name>.ops gets its own generated registration
# file + kernel source list, and the component's Kconfig choice selects
# which one compiles. Switching profiles never needs regeneration.
# A profile may also list cadence:: ops; those are codegen'd against
# backends/cadence/aot/functions.yaml and bound to the Cadence backend's
# generic (portable) kernels under backends/cadence/generic.
#
# Host requirements: python3 with ExecuTorch + torchgen importable (for
# kernel codegen), cmake (to build flatc from third-party/flatbuffers
# when no flatc is available), and git submodules third-party/flatcc +
# third-party/flatbuffers initialized. Consumers of the output need
# only ESP-IDF. Set PYTHON=... / FLATC=... to override discovery.
#
# NOTE: This script is coupled to the ExecuTorch source tree layout, but
# its source file lists come from the same build_variables.bzl the
# CMake build uses, so upstream file moves are picked up automatically.
# Consumer builds for Xtensa + RISC-V targets are the safety net.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ET_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEMPLATE_DIR="$SCRIPT_DIR/idf_component_template"
ESP_PAL_SRC="$SCRIPT_DIR/platform/esp_pal.cpp"
PYTHON="${PYTHON:-python3}"
PACKAGING_REV=3 # bump when the packaging (not ExecuTorch) changes

PROFILES_CSV=""
OUT_DIR=""
CLEAN=false

usage() {
  echo "Usage: $0 --profiles <csv> --out <dir> [--clean]" >&2
  echo "       $0 --version" >&2
  echo "Profiles are defined in $SCRIPT_DIR/idf_profiles/<name>.ops" >&2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --profiles) PROFILES_CSV="$2"; shift 2 ;;
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

if [ -z "$PROFILES_CSV" ]; then
  echo "ERROR: --profiles is required (e.g. --profiles wakeword_int8)." >&2
  usage; exit 1
fi
IFS=',' read -r -a PROFILES <<< "$PROFILES_CSV"
for p in "${PROFILES[@]}"; do
  if [ ! -f "$SCRIPT_DIR/idf_profiles/$p.ops" ]; then
    echo "ERROR: unknown profile '$p' (no idf_profiles/$p.ops)." >&2
    exit 1
  fi
done

ET_VERSION=$(cat "$ET_ROOT/version.txt")
if [[ ! "$ET_VERSION" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]; then
  echo "ERROR: version.txt must contain an ExecuTorch version in X.Y.Z form." >&2
  exit 1
fi
ET_COMPONENT_VERSION="${ET_VERSION}~${PACKAGING_REV}"

for dep in flatbuffers flatcc; do
  if [ ! -f "$ET_ROOT/third-party/$dep/LICENSE" ]; then
    echo "ERROR: third-party/$dep is empty. Run: git submodule update --init third-party/$dep" >&2
    exit 1
  fi
done

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
echo "  Profiles:  ${PROFILES[*]}"
echo "  Output:    $OUT_DIR"

# Print one source list from build/build_variables.bzl (same mechanism
# as tools/cmake/Codegen.cmake:executorch_append_filelist).
bzl_list() {
  "$PYTHON" -c "
exec(open('$ET_ROOT/shim_et/xplat/executorch/build/build_variables.bzl').read())
for f in $1: print(f)
"
}

# Print the sorted, comma-joined operator names of a selected_operators.yaml.
sel_ops_of_yaml() {
  "$PYTHON" - "$1" <<'PYEOF'
import sys
in_ops = False
ops = []
for line in open(sys.argv[1]):
    if line.startswith("operators:"):
        in_ops = True
        continue
    if in_ops:
        if line.startswith("  ") and not line.startswith("    ") and ":" in line:
            key = line.strip()
            ops.append(key[:-1] if key.endswith(":") else key)
        elif line and not line[0].isspace():
            break
print(",".join(sorted(ops)))
PYEOF
}

SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
mkdir -p "$SCRATCH/codegen"

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
         "$OUT_DIR/src/executorch" "$OUT_DIR/src/profiles"

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
echo "[1/6] Schema headers generated"

# ─────────────────────────────────────────────────────────
# 2. Per-profile kernel registration codegen.
#    Same recipe as the CMake selective build for portable kernels
#    (gen_oplist --root_ops + codegen.gen over kernels/portable/
#    functions.yaml), run once per profile.
# ─────────────────────────────────────────────────────────
if ! TORCHGEN=$("$PYTHON" -c "import torchgen, os; print(os.path.dirname(torchgen.__file__))" 2>/dev/null); then
  echo "ERROR: cannot import torchgen with $PYTHON." >&2
  echo "       Kernel codegen needs it. Run ./install_executorch.sh, or set" >&2
  echo "       PYTHON=... to an interpreter that has ExecuTorch installed." >&2
  exit 1
fi
for yaml in "$TORCHGEN/packaged/ATen/native/tags.yaml" \
            "$TORCHGEN/packaged/ATen/native/native_functions.yaml"; do
  if [ ! -f "$yaml" ]; then
    echo "ERROR: $yaml is missing from the torchgen at $TORCHGEN." >&2
    exit 1
  fi
done

# The exporter and the runtime must be the same ExecuTorch, or models
# fail at Method::execute with errors that do not say why. Warn loudly
# when the installed Python package was built from another commit.
ET_PY_SHA=$("$PYTHON" -c "import executorch.version as v; print(getattr(v, 'git_version', ''))" 2>/dev/null || echo "")
ET_GIT_SHA=$(git -C "$ET_ROOT" rev-parse HEAD 2>/dev/null || echo "")
if [ -z "$ET_PY_SHA" ]; then
  echo "  NOTE: no executorch Python package found for $PYTHON; codegen may still work."
elif [ -n "$ET_GIT_SHA" ] && [ "$ET_PY_SHA" != "$ET_GIT_SHA" ]; then
  echo "  WARNING: $PYTHON has executorch built from ${ET_PY_SHA:0:12}, this tree"
  echo "           is at ${ET_GIT_SHA:0:12}. Models exported with it may not match"
  echo "           this component. Run ./install_executorch.sh from this checkout."
fi

mkdir -p "$SCRATCH/kernel_union"
: > "$SCRATCH/kernel_union/files.txt"
: > "$SCRATCH/version_profiles.txt"
: > "$SCRATCH/profile_table.md"

for p in "${PROFILES[@]}"; do
  OPS_FILE="$SCRIPT_DIR/idf_profiles/$p.ops"
  ROOT_OPS=$(grep -v '^[[:space:]]*#' "$OPS_FILE" | grep -v '^[[:space:]]*$' \
             | tr '\n' ',' | sed 's/,$//')
  NROOT=$(echo "$ROOT_OPS" | tr ',' '\n' | wc -l | tr -d ' ')
  # A profile may mix aten:: ops (portable kernels) with cadence:: ops
  # (Cadence backend generic/portable kernels); each namespace gets its
  # own codegen run + registration file below.
  ATEN_OPS=$(echo "$ROOT_OPS" | tr ',' '\n' | grep -v '^cadence::' \
             | tr '\n' ',' | sed 's/,$//')
  CADENCE_OPS=$(echo "$ROOT_OPS" | tr ',' '\n' | grep '^cadence::' \
             | tr '\n' ',' | sed 's/,$//' || true)
  GEN_DIR="$SCRATCH/codegen/$p"
  mkdir -p "$GEN_DIR"

  ( cd "$ET_ROOT" && \
    "$PYTHON" -m codegen.tools.gen_oplist \
      --output_path="$GEN_DIR/selected_operators.yaml" \
      --root_ops="$ATEN_OPS" && \
    "$PYTHON" -m codegen.gen \
      --source-path="$ET_ROOT/codegen" \
      --install-dir="$GEN_DIR" \
      --tags-path="$TORCHGEN/packaged/ATen/native/tags.yaml" \
      --aten-yaml-path="$TORCHGEN/packaged/ATen/native/native_functions.yaml" \
      --op-selection-yaml-path="$GEN_DIR/selected_operators.yaml" \
      --functions-yaml-path="$ET_ROOT/kernels/portable/functions.yaml" \
  ) > "$GEN_DIR/codegen.log" 2>&1 || {
    echo "ERROR: codegen failed for profile '$p'; log follows:" >&2
    cat "$GEN_DIR/codegen.log" >&2; trap - EXIT; exit 1; }
  # Same content under two names; keeping both is a duplicate symbol.
  rm -f "$GEN_DIR/RegisterCodegenUnboxedKernels_0.cpp"

  if [ ! -f "$GEN_DIR/RegisterCodegenUnboxedKernelsEverything.cpp" ]; then
    echo "ERROR: codegen produced no registration file for profile '$p'." >&2
    exit 1
  fi
  NKERNELS=$(grep -c 'Kernel(' "$GEN_DIR/RegisterCodegenUnboxedKernelsEverything.cpp")

  # Cadence backend ops (cadence::): same two-step codegen against
  # backends/cadence/aot/functions.yaml, whose generic kernel bindings
  # point at the portable implementations under backends/cadence/generic
  # (the same recipe as Codegen.cmake gen_selected_ops +
  # generate_bindings_for_kernels). gen_oplist with the Cadence schema
  # yaml marks every op in that file as a root (97 ops, mostly aten
  # entries this component already registers from kernels/portable), so
  # the selection is filtered down to the profile's cadence:: roots.
  CAD_NKERNELS=0
  CADENCE_YAML=""
  if [ -n "$CADENCE_OPS" ]; then
    CADENCE_YAML_FILE="$ET_ROOT/backends/cadence/aot/functions.yaml"
    CAD_DIR="$GEN_DIR/cadence"
    mkdir -p "$CAD_DIR"
    ( cd "$ET_ROOT" && \
      "$PYTHON" -m codegen.tools.gen_oplist \
        --output_path="$CAD_DIR/selected_operators_unfiltered.yaml" \
        --ops_schema_yaml_path="$CADENCE_YAML_FILE" \
        --root_ops="$CADENCE_OPS" \
    ) > "$CAD_DIR/codegen.log" 2>&1 || {
      echo "ERROR: cadence gen_oplist failed for profile '$p'; log follows:" >&2
      cat "$CAD_DIR/codegen.log" >&2; trap - EXIT; exit 1; }
    if ! "$PYTHON" - "$CAD_DIR/selected_operators_unfiltered.yaml" \
        "$CAD_DIR/selected_operators.yaml" "$CADENCE_OPS" \
        >> "$CAD_DIR/codegen.log" 2>&1 <<'PYEOF'
import sys
import yaml
keep = set(sys.argv[3].split(","))
d = yaml.safe_load(open(sys.argv[1]))
d["operators"] = {k: v for k, v in d["operators"].items() if k in keep}
d["et_kernel_metadata"] = {
    k: v for k, v in d.get("et_kernel_metadata", {}).items() if k in keep
}
missing = keep - set(d["operators"])
if missing:
    sys.exit(f"cadence root ops not in functions.yaml: {sorted(missing)}")
yaml.safe_dump(d, open(sys.argv[2], "w"), sort_keys=True)
PYEOF
    then
      echo "ERROR: cadence op selection filter failed for profile '$p'; log follows:" >&2
      cat "$CAD_DIR/codegen.log" >&2; trap - EXIT; exit 1
    fi
    ( cd "$ET_ROOT" && \
      "$PYTHON" -m codegen.gen \
        --source-path="$ET_ROOT/codegen" \
        --install-dir="$CAD_DIR" \
        --tags-path="$TORCHGEN/packaged/ATen/native/tags.yaml" \
        --aten-yaml-path="$TORCHGEN/packaged/ATen/native/native_functions.yaml" \
        --op-selection-yaml-path="$CAD_DIR/selected_operators.yaml" \
        --functions-yaml-path="$CADENCE_YAML_FILE" \
    ) >> "$CAD_DIR/codegen.log" 2>&1 || {
      echo "ERROR: cadence codegen failed for profile '$p'; log follows:" >&2
      cat "$CAD_DIR/codegen.log" >&2; trap - EXIT; exit 1; }
    rm -f "$CAD_DIR/RegisterCodegenUnboxedKernels_0.cpp"
    if [ ! -f "$CAD_DIR/RegisterCodegenUnboxedKernelsEverything.cpp" ]; then
      echo "ERROR: cadence codegen produced no registration file for profile '$p'." >&2
      exit 1
    fi
    CAD_NKERNELS=$(grep -c 'Kernel(' "$CAD_DIR/RegisterCodegenUnboxedKernelsEverything.cpp")
    NKERNELS=$((NKERNELS + CAD_NKERNELS))
    CADENCE_YAML="$CAD_DIR/selected_operators.yaml"
  fi

  ( cd "$ET_ROOT" && "$PYTHON" -m codegen.tools.gen_max_kernel_num \
      --oplist-yaml="$GEN_DIR/selected_operators.yaml" \
      ${CADENCE_YAML:+--oplist-yaml="$CADENCE_YAML"} \
      --prim-ops-source="$ET_ROOT/kernels/prim_ops/register_prim_ops.cpp" \
      --output-path="$GEN_DIR/selected_max_kernel_num.h" ) >/dev/null 2>&1
  MAXK=$(sed -n 's/.*EXECUTORCH_SELECTED_MAX_KERNEL_NUM \([0-9]*\).*/\1/p' \
           "$GEN_DIR/selected_max_kernel_num.h")
  if [ -z "$MAXK" ]; then
    echo "ERROR: could not determine MAX_KERNEL_NUM for profile '$p'." >&2
    exit 1
  fi

  # Effective op set + stable hash, from selected_operators.yaml.
  SEL_OPS=$(sel_ops_of_yaml "$GEN_DIR/selected_operators.yaml")
  if [ -n "$CADENCE_YAML" ]; then
    SEL_OPS=$(printf '%s,%s' "$SEL_OPS" "$(sel_ops_of_yaml "$CADENCE_YAML")" \
              | tr ',' '\n' | sort -u | tr '\n' ',' | sed 's/,$//')
  fi
  OP_HASH=$(printf '%s' "$SEL_OPS" | "$PYTHON" -c \
    "import hashlib,sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())")

  # Map each referenced kernel function to the portable .cpp defining
  # it. Over-inclusion is safe (link-time GC drops the rest); a kernel
  # that maps to no file is a hard error here rather than on device.
  : > "$GEN_DIR/kernel_files.txt"
  MISSING=""
  for fn in $(grep -o 'native::[A-Za-z_0-9]*' \
                "$GEN_DIR/RegisterCodegenUnboxedKernelsEverything.cpp" \
              | sed 's/native:://' | sort -u); do
    found=false
    while IFS= read -r f; do
      if grep -q "namespace native" "$f"; then
        echo "${f#"$ET_ROOT"/}" >> "$GEN_DIR/kernel_files.txt"
        found=true
      fi
    # Match plain definitions `fn(` and macro definitions
    # DEFINE_..._UFUNC(fn, ...) used by the unary/binary patterns.
    done < <(grep -rlE "(^|[^A-Za-z0-9_])${fn}[(,]" \
               "$ET_ROOT/kernels/portable/cpu" --include='*.cpp' || true)
    if [ "$found" = false ]; then MISSING="$MISSING $fn"; fi
  done
  # Macro/pattern-generated kernels live here and evade the name grep.
  find "$ET_ROOT/kernels/portable/cpu/pattern" -name '*.cpp' \
    | sed "s|^$ET_ROOT/||" >> "$GEN_DIR/kernel_files.txt"
  if [ -n "$CADENCE_OPS" ]; then
    for fn in $(grep -o 'native::[A-Za-z_0-9]*' \
                  "$CAD_DIR/RegisterCodegenUnboxedKernelsEverything.cpp" \
                | sed 's/native:://' | sort -u); do
      found=false
      while IFS= read -r f; do
        if grep -q "namespace native" "$f"; then
          echo "${f#"$ET_ROOT"/}" >> "$GEN_DIR/kernel_files.txt"
          found=true
        fi
      done < <(grep -rlE "(^|[^A-Za-z0-9_])${fn}[(,]" \
                 "$ET_ROOT/backends/cadence/generic" --include='*.cpp' || true)
      if [ "$found" = false ]; then MISSING="$MISSING cadence:$fn"; fi
    done
    # Shared quantize/requantize helpers the generic operators call.
    echo "backends/cadence/generic/kernels/kernels.cpp" \
      >> "$GEN_DIR/kernel_files.txt"
  fi
  sort -u "$GEN_DIR/kernel_files.txt" -o "$GEN_DIR/kernel_files.txt"
  if [ -n "$MISSING" ]; then
    echo "ERROR: profile '$p': no portable kernel source found for:$MISSING" >&2
    echo "       (they may be macro-generated in an unexpected file; extend the mapping above)" >&2
    exit 1
  fi

  mkdir -p "$OUT_DIR/src/profiles/$p"
  cp "$GEN_DIR/RegisterCodegenUnboxedKernelsEverything.cpp" \
     "$OUT_DIR/src/profiles/$p/register_kernels.cpp"
  cp "$GEN_DIR/NativeFunctions.h" "$GEN_DIR/Functions.h" \
     "$OUT_DIR/src/profiles/$p/"
  {
    echo "src/profiles/$p/register_kernels.cpp"
    if [ -n "$CADENCE_OPS" ]; then
      # The cadence registration file includes its own NativeFunctions.h
      # (same file names as the aten ones, different contents), so it
      # gets its own directory, like a second generated library.
      mkdir -p "$OUT_DIR/src/profiles/$p/cadence"
      cp "$CAD_DIR/RegisterCodegenUnboxedKernelsEverything.cpp" \
         "$OUT_DIR/src/profiles/$p/cadence/register_kernels.cpp"
      cp "$CAD_DIR/NativeFunctions.h" "$CAD_DIR/Functions.h" \
         "$OUT_DIR/src/profiles/$p/cadence/"
      echo "src/profiles/$p/cadence/register_kernels.cpp"
    fi
  } > "$GEN_DIR/profile_srcs.txt"

  cat "$GEN_DIR/kernel_files.txt" >> "$SCRATCH/kernel_union/files.txt"
  {
    echo "  $p:"
    echo "    root_ops: $NROOT (idf_profiles/$p.ops)"
    echo "    registered_kernels: $NKERNELS"
    echo "    max_kernel_num: $MAXK"
    echo "    op_set_sha256: $OP_HASH"
    echo "    selected_ops: $SEL_OPS"
  } >> "$SCRATCH/version_profiles.txt"
  echo "| \`$p\` | $NROOT root ops, $NKERNELS kernels | \`${OP_HASH:0:12}...\` |" \
    >> "$SCRATCH/profile_table.md"
  echo "$MAXK" > "$GEN_DIR/max_kernel_num.txt"
  echo "  profile $p: $NROOT root ops -> $NKERNELS kernels (registry $MAXK)"
done
sort -u "$SCRATCH/kernel_union/files.txt" -o "$SCRATCH/kernel_union/files.txt"
echo "[2/6] Kernel registration generated for: ${PROFILES[*]}"

# ─────────────────────────────────────────────────────────
# 3. Vendor runtime + extension sources (build_variables.bzl
#    lists) and the union of profile kernel sources.
#    Output mirrors upstream paths under src/executorch/.
# ─────────────────────────────────────────────────────────
: > "$SCRATCH/core_srcs.txt"

copy_bzl_list() { # $1=bzl variable name; appends to core_srcs.txt
  local f
  bzl_list "$1" | while IFS= read -r f; do
    if [ ! -f "$ET_ROOT/$f" ]; then
      echo "ERROR: build_variables.bzl lists missing file: $f" >&2; exit 1
    fi
    mkdir -p "$OUT_DIR/src/executorch/$(dirname "$f")"
    cp "$ET_ROOT/$f" "$OUT_DIR/src/executorch/$f"
    echo "src/executorch/$f" >> "$SCRATCH/core_srcs.txt"
  done
}

copy_bzl_list EXECUTORCH_CORE_SRCS
copy_bzl_list EXECUTORCH_SRCS # prim ops
copy_bzl_list EXTENSION_MODULE_SRCS
copy_bzl_list EXTENSION_FLAT_TENSOR_SRCS
copy_bzl_list EXTENSION_NAMED_DATA_MAP_SRCS
copy_bzl_list EXTENSION_MEMORY_ALLOCATOR_SRCS
copy_bzl_list KERNELS_UTIL_ALL_DEPS_SRCS

# Data loaders: buffer loader is header-only; file loader works over
# VFS; the mmap loader is replaced by the ESP-IDF stub (no mmap).
mkdir -p "$OUT_DIR/src/executorch/extension/data_loader"
cp "$ET_ROOT/extension/data_loader/file_data_loader.cpp" \
   "$OUT_DIR/src/executorch/extension/data_loader/"
cp "$TEMPLATE_DIR/src/data_loader/mmap_data_loader.cpp" \
   "$OUT_DIR/src/executorch/extension/data_loader/mmap_data_loader.cpp"
{
  echo "src/executorch/extension/data_loader/file_data_loader.cpp"
  echo "src/executorch/extension/data_loader/mmap_data_loader.cpp"
} >> "$SCRATCH/core_srcs.txt"

# Kernel sources needed by at least one shipped profile.
: > "$SCRATCH/kernel_srcs_by_profile.txt"
while IFS= read -r f; do
  mkdir -p "$OUT_DIR/src/executorch/$(dirname "$f")"
  cp "$ET_ROOT/$f" "$OUT_DIR/src/executorch/$f"
done < "$SCRATCH/kernel_union/files.txt"

echo "[3/6] Runtime, extension and kernel sources copied"

# ─────────────────────────────────────────────────────────
# 4. Vendor headers: executorch/** (public), c10/torch,
#    flatbuffers, flatcc (+ its C runtime).
# ─────────────────────────────────────────────────────────
copy_headers() { # $1=src dir under ET_ROOT, $2=dest under OUT/include
  local src_dir="$1" dest_dir="$2" f
  find "$ET_ROOT/$src_dir" \( -name '*.h' -o -name '*.hpp' \) \
      -not -path '*/test/*' | while IFS= read -r f; do
    local rel="${f#"$ET_ROOT/$src_dir"/}"
    mkdir -p "$OUT_DIR/include/$dest_dir/$(dirname "$rel")"
    cp "$f" "$OUT_DIR/include/$dest_dir/$rel"
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
for d in module data_loader flat_tensor named_data_map memory_allocator; do
  copy_headers "extension/$d" "executorch/extension/$d"
done
copy_headers kernels/portable executorch/kernels/portable
copy_headers kernels/prim_ops executorch/kernels/prim_ops

# Cadence backend generic (portable) kernel headers, only when some
# shipped profile selects cadence:: ops.
ANY_CADENCE=false
for p in "${PROFILES[@]}"; do
  if grep -q '^[[:space:]]*cadence::' "$SCRIPT_DIR/idf_profiles/$p.ops"; then
    ANY_CADENCE=true
  fi
done
if [ "$ANY_CADENCE" = true ]; then
  copy_headers backends/cadence/generic executorch/backends/cadence/generic
fi

cp -r "$ET_ROOT/third-party/flatbuffers/include/flatbuffers" \
      "$OUT_DIR/include/flatbuffers"
mkdir -p "$OUT_DIR/include/flatcc" "$OUT_DIR/src/flatcc/runtime"
cp "$ET_ROOT/third-party/flatcc/include/flatcc/"*.h "$OUT_DIR/include/flatcc/"
cp -r "$ET_ROOT/third-party/flatcc/include/flatcc/portable" \
      "$OUT_DIR/include/flatcc/"
cp "$ET_ROOT/third-party/flatcc/src/runtime/"*.c "$OUT_DIR/src/flatcc/runtime/"
for f in "$OUT_DIR/src/flatcc/runtime/"*.c; do
  echo "src/flatcc/runtime/$(basename "$f")" >> "$SCRATCH/core_srcs.txt"
done

echo "[4/6] Headers and third-party code copied"

# ─────────────────────────────────────────────────────────
# 5. Component glue from idf_component_template/ and the shared
#    ESP-IDF PAL, plus generated metadata (et_generated.cmake,
#    Kconfig, idf_component.yml, README, VERSION, licenses).
# ─────────────────────────────────────────────────────────
cp "$TEMPLATE_DIR/CMakeLists.txt" "$OUT_DIR/CMakeLists.txt"
cp "$TEMPLATE_DIR/include/et_model.h" "$OUT_DIR/include/"
mkdir -p "$OUT_DIR/src/platform"
cp "$TEMPLATE_DIR/src/et_model.cpp" "$OUT_DIR/src/"
cp "$ESP_PAL_SRC" "$OUT_DIR/src/platform/"
cp "$ET_ROOT/LICENSE" "$OUT_DIR/LICENSE"

LICENSES="$OUT_DIR/THIRD_PARTY_LICENSES"
mkdir -p "$LICENSES"
for dep in flatbuffers flatcc; do
  cp "$ET_ROOT/third-party/$dep/LICENSE" "$LICENSES/$dep-LICENSE.txt"
done
cat > "$LICENSES/README.md" <<'NOTICE'
# Third-party licenses

This component redistributes source from the projects below.
ExecuTorch's own BSD license is in the LICENSE file at the root.

| Component | Location in this component | License |
|---|---|---|
| FlatBuffers (Google) | `include/flatbuffers/` | Apache-2.0 — `flatbuffers-LICENSE.txt` |
| flatcc (Mikkel F. Jorgensen) | `include/flatcc/`, `src/flatcc/` | Apache-2.0 — `flatcc-LICENSE.txt` |
| PyTorch c10 (Meta) | `include/c10/`, `include/torch/` | BSD-3-Clause, as exact copies from PyTorch core |

`src/profiles/*/register_kernels.cpp` is generated by ExecuTorch's
codegen from PyTorch's `native_functions.yaml` and carries the same
terms as ExecuTorch itself.
NOTICE

# et_generated.cmake: baked source lists consumed by CMakeLists.txt.
{
  echo "# GENERATED by examples/espressif/build_idf_component.sh -- do not edit."
  echo "set(ET_PROFILES ${PROFILES[*]})"
  echo "set(ET_DEFAULT_PROFILE ${PROFILES[0]})"
  echo "set(ET_CORE_SRCS"
  sort -u "$SCRATCH/core_srcs.txt" | sed 's/.*/    "&"/'
  echo ")"
  for p in "${PROFILES[@]}"; do
    up=$(echo "$p" | tr '[:lower:]-' '[:upper:]_')
    echo "set(ET_PROFILE_SRCS_$up"
    sed 's|.*|    "&"|' "$SCRATCH/codegen/$p/profile_srcs.txt"
    echo ")"
    echo "set(ET_KERNEL_SRCS_$up"
    sed 's|.*|    "src/executorch/&"|' "$SCRATCH/codegen/$p/kernel_files.txt"
    echo ")"
    echo "set(ET_MAX_KERNEL_NUM_$up $(cat "$SCRATCH/codegen/$p/max_kernel_num.txt"))"
  done
} > "$OUT_DIR/et_generated.cmake"

# Kconfig: profile choice generated from the shipped profiles.
{
  cat <<'KHEAD'
# GENERATED by examples/espressif/build_idf_component.sh -- do not edit.
menu "ExecuTorch"

    config ET_ENABLE
        bool "Build the ExecuTorch runtime into the firmware"
        default y
        help
            ESP-IDF links every component under components/ into the
            image, and this component's kernel registration is
            deliberately GC-proof (whole-archive). Projects that
            vendor the component before using it set this to n (the
            component then registers empty and firmware is unchanged);
            a consumer's Kconfig can also "select ET_ENABLE".

    if ET_ENABLE

    choice ET_OP_PROFILE
        prompt "Op profile"
        help
            Which operator set (kernel registration) to compile in.
            Baked in at component generation time; see VERSION for the
            exact ops and op-set hash of each profile. A .pte only
            runs on a profile whose op set covers it.
KHEAD
  first=true
  for p in "${PROFILES[@]}"; do
    up=$(echo "$p" | tr '[:lower:]-' '[:upper:]_')
    if [ "$first" = true ]; then
      echo "        default ET_OP_PROFILE_$up"
      first=false
    fi
    echo "        config ET_OP_PROFILE_$up"
    echo "            bool \"$p\""
  done
  cat <<'KTAIL'
    endchoice

    config ET_ENABLE_LOGGING
        bool "Enable ExecuTorch runtime logging"
        default y
        help
            When disabled, ET_LOG is compiled out entirely
            (ET_LOG_ENABLED=0). When enabled, messages go through
            ESP_LOG under the "executorch" tag.

    choice ET_LOG_LEVEL
        prompt "ExecuTorch log level floor"
        default ET_LOG_LEVEL_INFO
        depends on ET_ENABLE_LOGGING
        help
            Compile-time floor applied in the PAL on top of the
            usual IDF runtime log levels.

        config ET_LOG_LEVEL_ERROR
            bool "Error"
        config ET_LOG_LEVEL_WARN
            bool "Warning"
        config ET_LOG_LEVEL_INFO
            bool "Info"
        config ET_LOG_LEVEL_DEBUG
            bool "Debug"
    endchoice

    endif # ET_ENABLE

endmenu
KTAIL
} > "$OUT_DIR/Kconfig"

sed "s/@ET_COMPONENT_VERSION@/$ET_COMPONENT_VERSION/" \
  "$TEMPLATE_DIR/idf_component.yml.in" > "$OUT_DIR/idf_component.yml"

PROFILES_HUMAN=$(IFS=','; echo "${PROFILES[*]}")
{
  echo "| Profile | Ops | Op-set SHA-256 |"
  echo "|---|---|---|"
  cat "$SCRATCH/profile_table.md"
} > "$SCRATCH/profile_table_full.md"
"$PYTHON" - "$TEMPLATE_DIR/README.md" "$SCRATCH/profile_table_full.md" \
  "$PROFILES_HUMAN" > "$OUT_DIR/README.md" <<'PYEOF'
import sys
readme = open(sys.argv[1]).read()
table = open(sys.argv[2]).read().rstrip("\n")
readme = readme.replace("@PROFILE_TABLE@", table)
readme = readme.replace("@PROFILES@", sys.argv[3])
sys.stdout.write(readme)
PYEOF

ET_SHA=$(git -C "$ET_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")
ET_DIRTY=$(git -C "$ET_ROOT" status --porcelain 2>/dev/null | head -1)
{
  echo "# GENERATED by examples/espressif/build_idf_component.sh -- do not edit."
  echo "executorch_version: $ET_VERSION"
  echo "component_version: $ET_COMPONENT_VERSION"
  echo "executorch_commit: $ET_SHA${ET_DIRTY:+ (tree had uncommitted changes)}"
  echo "generator: examples/espressif/build_idf_component.sh --profiles $PROFILES_CSV --out $OUT_DIR_AS_GIVEN"
  echo "profiles:"
  cat "$SCRATCH/version_profiles.txt"
  echo "substitutions:"
  echo "  - extension/data_loader/mmap_data_loader.cpp: ESP-IDF stub (no mmap; returns NotSupported)"
  echo "  - torch/headeronly/macros/cmake_macros.h: packaging stub (as in examples/arduino)"
  echo "notes:"
  echo "  - application pipelines (for example wake word) are not included; they live with the application"
  echo "  - no .pte models are included; models ship with the application component"
  if [ "$ANY_CADENCE" = true ]; then
    echo "  - cadence:: ops use the Cadence backend's generic (portable) kernels (backends/cadence/generic), not the HiFi/xt-clang kernels"
  fi
} > "$OUT_DIR/VERSION"

echo "[5/6] Component glue and metadata written"

# ─────────────────────────────────────────────────────────
# 6. Sanity checks + summary
# ─────────────────────────────────────────────────────────
for f in "CMakeLists.txt" "et_generated.cmake" "Kconfig" \
         "idf_component.yml" "VERSION" "LICENSE" "include/et_model.h" \
         "include/executorch/runtime/executor/program.h" \
         "include/executorch/schema/program_generated.h" \
         "src/et_model.cpp" "src/platform/esp_pal.cpp"; do
  if [ ! -f "$OUT_DIR/$f" ]; then
    echo "ERROR: generated component is missing $f" >&2; exit 1
  fi
done
for p in "${PROFILES[@]}"; do
  if [ ! -f "$OUT_DIR/src/profiles/$p/register_kernels.cpp" ]; then
    echo "ERROR: generated component is missing profile '$p' registration" >&2
    exit 1
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

echo "[6/6] Done"
echo ""
echo "=== ExecuTorch ESP-IDF component $ET_COMPONENT_VERSION ==="
echo "  Output:    $OUT_DIR ($SIZE)"
echo "  Sources:   $NSRC, headers: $NHDR"
echo "  Profiles:  ${PROFILES[*]}"
echo ""
echo "Next: build a consumer ESP-IDF project that vendors this under"
echo "components/executorch, select a profile (CONFIG_ET_OP_PROFILE_*),"
echo "and build for your target, e.g. idf.py set-target esp32s3 build."
