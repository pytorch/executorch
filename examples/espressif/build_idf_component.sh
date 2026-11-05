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
#   ./build_idf_component.sh --out /path/to/components/executorch [stamp]
#   ./build_idf_component.sh --version [stamp]
#   ./build_idf_component.sh --out /path/to/components/executorch --clean
#
# The component version is version.txt's X.Y.Z plus a suffix, which the
# stamp options select and check against git:
#   (none) or --version-suffix ""  X.Y.Z; HEAD must be at the vX.Y.Z tag.
#   --version-suffix -rc.N         X.Y.Z-rc.N; HEAD must be at vX.Y.Z-rcN.
#   --version-suffix '~N'          X.Y.Z~N, a respin of the component (its
#                                  packaging, PAL and C API): HEAD must be on
#                                  release/X.Y, after vX.Y.Z, and differ from
#                                  it only under examples/espressif/, .ci/,
#                                  .github/ and docs/.
#   --allow-dev-stamp              X.Y.Z-dev.<HEAD's commit date>, for builds
#                                  from any commit; never published.
#
# Derived from examples/arduino/build_arduino_library.sh with an ESP-IDF
# output layout and the shared ESP-IDF PAL. The component vendors every
# kernel source of its kernel libraries (portable, quantized, the Cadence
# backend's generic kernels and the esp_nn:: op library) and the ESP-NN
# delegate, but registers none of them: a
# project's kernel bundle, made by the component's tools/gen_kernels.py
# from its models, selects the kernels at build time.
#
# Host requirements: a Python with the executorch package built from this
# checkout (to read build_variables.bzl, and to export the shipped
# example's model and make its kernel bundle), cmake (to build flatc from
# third-party/flatbuffers when no flatc is available), and the git
# submodule third-party/flatbuffers initialized. Consumers of the output
# need only ESP-IDF, plus an ExecuTorch Python environment to make kernel
# bundles. Set PYTHON=... / FLATC=... to override discovery.
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

OUT_DIR=""
CLEAN=false
PRINT_VERSION=false
VERSION_SUFFIX=""
VERSION_SUFFIX_GIVEN=false
ALLOW_DEV_STAMP=false

usage() {
  echo "Usage: $0 --out <dir> [--version-suffix <suffix> | --allow-dev-stamp] [--clean]" >&2
  echo "       $0 --version [--version-suffix <suffix> | --allow-dev-stamp]" >&2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT_DIR="$2"; shift 2 ;;
    --clean) CLEAN=true; shift ;;
    --version) PRINT_VERSION=true; shift ;;
    --version-suffix) VERSION_SUFFIX="$2"; VERSION_SUFFIX_GIVEN=true; shift 2 ;;
    --allow-dev-stamp) ALLOW_DEV_STAMP=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

OUT_DIR_AS_GIVEN="$OUT_DIR"
# Normalize to an absolute path (dir may not exist yet).
case "$OUT_DIR" in
  ""|/*) ;;
  *) OUT_DIR="$PWD/$OUT_DIR" ;;
esac

if [ "$CLEAN" = true ]; then
  if [ -z "$OUT_DIR" ]; then
    echo "ERROR: --out is required." >&2; usage; exit 1
  fi
  if [ -e "$OUT_DIR/et_generated.cmake" ] || [ -e "$OUT_DIR/.et-generated-component" ]; then
    rm -rf "$OUT_DIR"
    echo "Cleaned $OUT_DIR"
  else
    echo "Nothing to clean: $OUT_DIR is not a generated component." >&2
  fi
  exit 0
fi

stamp_error() {
  echo "ERROR: $1" >&2
  exit 1
}

# The trees the generator copies files from by globbing.
KERNEL_SRC_ROOTS=(kernels/portable/cpu kernels/quantized/cpu backends/cadence/generic
  backends/espressif/ops)
HEADER_ROOTS=(runtime schema kernels/portable kernels/quantized kernels/prim_ops
  backends/cadence/generic)
ESPNN_ROOT=backends/espressif/espnn_runtime
CODEGEN_TEMPLATES=codegen/templates
FLATBUFFERS=third-party/flatbuffers

# Changes that would reach the component: to tracked files, or untracked files
# where the generator globs, including in the flatbuffers submodule.
uncommitted_changes() {
  git -C "$ET_ROOT" status --porcelain --untracked-files=no
  git -C "$ET_ROOT" ls-files --others --exclude-standard -- "${KERNEL_SRC_ROOTS[@]}" \
    "${HEADER_ROOTS[@]}" "$ESPNN_ROOT" "$CODEGEN_TEMPLATES"
  if [ -e "$ET_ROOT/$FLATBUFFERS/.git" ]; then
    git -C "$ET_ROOT/$FLATBUFFERS" ls-files --others --exclude-standard -- include/flatbuffers \
      | sed "s|^|$FLATBUFFERS/|"
  fi
}

# An ExecuTorch tree inside another repository must not take that one's history.
if ! ET_SHA=$(git -C "$ET_ROOT" rev-parse -q --verify HEAD 2> /dev/null) \
   || [ -n "$(git -C "$ET_ROOT" rev-parse --show-prefix)" ]; then
  stamp_error "the version stamp reads ExecuTorch's git history; generate from a git checkout."
fi

ET_VERSION=$(cat "$ET_ROOT/version.txt")
if [[ ! "$ET_VERSION" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]; then
  stamp_error "version.txt must contain an ExecuTorch version in X.Y.Z form."
fi
if [ "$ALLOW_DEV_STAMP" = true ]; then
  if [ "$VERSION_SUFFIX_GIVEN" = true ]; then
    stamp_error "--allow-dev-stamp picks the suffix; drop --version-suffix."
  fi
  VERSION_SUFFIX="-dev.$(TZ=UTC git -C "$ET_ROOT" log -1 --format=%cd --date=format-local:%Y%m%d)"
fi
# A release commit can carry its last RC's tag too.
HEAD_TAGS=$(git -C "$ET_ROOT" tag --points-at HEAD 2>/dev/null || true)
head_has_tag() {
  grep -qxF "$1" <<< "$HEAD_TAGS"
}
case "$VERSION_SUFFIX" in
  "")
    if ! head_has_tag "v$ET_VERSION"; then
      stamp_error "a release stamp needs HEAD at the release tag v$ET_VERSION. Pass --version-suffix, or --allow-dev-stamp for a development build."
    fi
    ;;
  -rc.*)
    if [[ ! "$VERSION_SUFFIX" =~ ^-rc\.[1-9][0-9]*$ ]]; then
      stamp_error "--version-suffix $VERSION_SUFFIX is not -rc.N."
    fi
    if ! head_has_tag "v$ET_VERSION-rc${VERSION_SUFFIX#-rc.}"; then
      stamp_error "an RC stamp needs HEAD at the tag v$ET_VERSION-rc${VERSION_SUFFIX#-rc.}."
    fi
    ;;
  "~"*)
    if [[ ! "$VERSION_SUFFIX" =~ ^~[1-9][0-9]*$ ]]; then
      stamp_error "--version-suffix $VERSION_SUFFIX is not ~N with N >= 1."
    fi
    RELEASE_BRANCH="release/${ET_VERSION%.*}"
    ON_RELEASE_BRANCH=false
    for ref in "refs/heads/$RELEASE_BRANCH" "refs/remotes/origin/$RELEASE_BRANCH"; do
      if git -C "$ET_ROOT" rev-parse -q --verify "$ref" > /dev/null \
         && git -C "$ET_ROOT" merge-base --is-ancestor HEAD "$ref"; then
        ON_RELEASE_BRANCH=true
      fi
    done
    if [ "$ON_RELEASE_BRANCH" = false ]; then
      stamp_error "a respin stamp needs HEAD on $RELEASE_BRANCH."
    fi
    if ! git -C "$ET_ROOT" rev-parse -q --verify "refs/tags/v$ET_VERSION" > /dev/null; then
      stamp_error "a respin stamp needs the release tag v$ET_VERSION."
    fi
    if ! git -C "$ET_ROOT" merge-base --is-ancestor "v$ET_VERSION" HEAD; then
      stamp_error "a respin must come after the release tag v$ET_VERSION."
    fi
    if [ "$(git -C "$ET_ROOT" rev-parse "v$ET_VERSION^{commit}")" = "$ET_SHA" ]; then
      stamp_error "a respin needs commits after the release tag v$ET_VERSION; HEAD is the release."
    fi
    CHANGED=$(git -C "$ET_ROOT" diff --name-only "v$ET_VERSION" HEAD -- . ':(exclude)examples/espressif' \
      ':(exclude).ci' ':(exclude).github' ':(exclude)docs')
    if [ -n "$CHANGED" ]; then
      stamp_error "a respin may change only examples/espressif/, .ci/, .github/ and docs/ since v$ET_VERSION, but HEAD also changes:
$CHANGED"
    fi
    ;;
  *)
    # The only other suffix is the one --allow-dev-stamp picks.
    if [ "$ALLOW_DEV_STAMP" = false ]; then
      stamp_error "--version-suffix must be empty, -rc.N or ~N, not $VERSION_SUFFIX; --allow-dev-stamp makes a development stamp."
    fi
    ;;
esac
if [ "$ALLOW_DEV_STAMP" = false ]; then
  CHANGES=$(uncommitted_changes)
  if [ -n "$CHANGES" ]; then
    stamp_error "a release, RC or respin stamp needs a checkout without uncommitted changes to what the component ships, but found:
$CHANGES"
  fi
fi
ET_COMPONENT_VERSION="$ET_VERSION$VERSION_SUFFIX"
if [ "$ALLOW_DEV_STAMP" = true ]; then
  STAMP_OPTION=" --allow-dev-stamp"
elif [ "$VERSION_SUFFIX_GIVEN" = true ]; then
  STAMP_OPTION=" --version-suffix '$VERSION_SUFFIX'"
else
  STAMP_OPTION=""
fi
if [ "$PRINT_VERSION" = true ]; then
  echo "$ET_COMPONENT_VERSION"
  exit 0
fi

if [ -z "$OUT_DIR" ]; then
  echo "ERROR: --out is required." >&2; usage; exit 1
fi

# The shipped example's model and kernel bundle are made with the executorch
# package, so it must be the one built from this commit.
if ! INSTALLED_SHA=$("$PYTHON" -c 'import executorch.version as v; print(v.git_version)' 2> /dev/null); then
  INSTALLED_SHA="none ($("$PYTHON" -c 'import executorch.version' 2>&1 | tail -n 1 || true))"
fi
if [ "$INSTALLED_SHA" != "$ET_SHA" ]; then
  echo "ERROR: the generator exports examples/hello_executorch with the executorch package, which must be installed from this checkout at $ET_SHA (reinstall after committing or pulling); found $INSTALLED_SHA." >&2
  exit 1
fi

if [ ! -f "$ET_ROOT/$FLATBUFFERS/LICENSE" ]; then
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
  cmake -S "$ET_ROOT/$FLATBUFFERS" -B "$SCRATCH/flatc-build" \
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
echo "[1/6] Schema headers generated"

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
: > "$SCRATCH/espnn_srcs.txt"
bzl_list EXECUTORCH_CORE_SRCS | copy_srcs "$SCRATCH/core_srcs.txt"
# register_prim_ops.cpp is compiled by the build, with the bundle's prim
# selection, only when a project has a kernel bundle.
bzl_list EXECUTORCH_SRCS | grep -v '/register_prim_ops\.cpp$' \
  | copy_srcs "$SCRATCH/kernel_srcs.txt"
echo "kernels/prim_ops/register_prim_ops.cpp" | copy_srcs "$SCRATCH/prim_src.txt"
( cd "$ET_ROOT" && find "${KERNEL_SRC_ROOTS[@]}" -name '*.cpp' -not -path '*/test*/*' | sort ) \
  | copy_srcs "$SCRATCH/kernel_srcs.txt"
# The ESP-NN delegate runtime, an ESP-IDF component of its own in-tree.
ESPNN_RUNTIME="$ET_ROOT/$ESPNN_ROOT"
mkdir -p "$OUT_DIR/src/espnn"
for f in "$ESPNN_RUNTIME"/src/*.cpp; do
  cp "$f" "$OUT_DIR/src/espnn/"
  echo "src/espnn/$(basename "$f")" >> "$SCRATCH/espnn_srcs.txt"
done

echo "[2/6] Runtime, extension and kernel sources copied"

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

for root in "${HEADER_ROOTS[@]}"; do
  copy_headers "$root" "executorch/$root"
done
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

# Only the header-only BufferDataLoader; Module, the file/mmap loaders
# and .ptd data maps are not vendored.
mkdir -p "$OUT_DIR/include/executorch/extension/data_loader"
cp "$ET_ROOT/extension/data_loader/buffer_data_loader.h" \
   "$OUT_DIR/include/executorch/extension/data_loader/"
# register_prim_ops.cpp includes selective_build_prim_ops.h by its bare
# name, which resolves next to the source.
cp "$ET_ROOT/kernels/prim_ops/selective_build_prim_ops.h" \
   "$OUT_DIR/src/executorch/kernels/prim_ops/"

mkdir -p "$OUT_DIR/priv_include"
cp -r "$ET_ROOT/$FLATBUFFERS/include/flatbuffers" "$OUT_DIR/priv_include/flatbuffers"
# Only espnn_backend.h is the delegate's API; the rest is internal to it.
cp -r "$ESPNN_RUNTIME/include/espnn" "$OUT_DIR/priv_include/espnn"
mkdir -p "$OUT_DIR/include/espnn"
mv "$OUT_DIR/priv_include/espnn/espnn_backend.h" "$OUT_DIR/include/espnn/"

echo "[3/6] Headers and third-party code copied"

# ─────────────────────────────────────────────────────────
# 4. Kernel bundle tooling: gen_kernels.py, the kernel
#    libraries' yamls and ExecuTorch's codegen templates.
#    Library order is gen_kernels.py's assignment priority.
# ─────────────────────────────────────────────────────────
LIBRARIES=(
  "portable kernels/portable/functions.yaml"
  "quantized kernels/quantized/quantized.yaml"
  "cadence backends/cadence/aot/functions.yaml"
  "esp_nn backends/espressif/ops/operators.yaml"
)
mkdir -p "$OUT_DIR/tools/libraries" "$OUT_DIR/tools/codegen"
cp "$TEMPLATE_DIR/tools/gen_kernels.py" "$OUT_DIR/tools/"
cp -r "$ET_ROOT/$CODEGEN_TEMPLATES" "$OUT_DIR/tools/codegen/templates"
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

echo "[4/6] Kernel bundle tooling copied"

# ─────────────────────────────────────────────────────────
# 5. Component glue from idf_component_template/ and the shared
#    ESP-IDF PAL, plus generated metadata (et_generated.cmake,
#    Kconfig, idf_component.yml, VERSION, licenses).
# ─────────────────────────────────────────────────────────
cp "$TEMPLATE_DIR/CMakeLists.txt" "$TEMPLATE_DIR/project_include.cmake" "$OUT_DIR/"
sed "s/@ET_COMPONENT_VERSION@/$ET_COMPONENT_VERSION/" "$TEMPLATE_DIR/README.md" \
  > "$OUT_DIR/README.md"
cp "$TEMPLATE_DIR/include/et_model.h" "$OUT_DIR/include/"
mkdir -p "$OUT_DIR/src/platform"
cp "$TEMPLATE_DIR/src/et_model.cpp" "$TEMPLATE_DIR/src/espnn_anchor.cpp" "$OUT_DIR/src/"
cp "$ESP_PAL_SRC" "$OUT_DIR/src/platform/"
cp "$ET_ROOT/LICENSE" "$OUT_DIR/LICENSE"

LICENSES="$OUT_DIR/THIRD_PARTY_LICENSES"
mkdir -p "$LICENSES"
cp "$ET_ROOT/$FLATBUFFERS/LICENSE" "$LICENSES/flatbuffers-LICENSE.txt"
# LLVM's license, with the legacy NCSA license that array_ref.h names.
cp "$TEMPLATE_DIR/THIRD_PARTY_LICENSES/llvm-LICENSE.TXT" "$LICENSES/"
cat > "$LICENSES/README.md" <<'NOTICE'
# Third-party licenses

This component redistributes source from the projects below.
ExecuTorch's own BSD license is in the LICENSE file at the root.

| Component | Location in this component | License |
|---|---|---|
| FlatBuffers (Google) | `priv_include/flatbuffers/` | Apache-2.0 — `flatbuffers-LICENSE.txt` |
| PyTorch c10 (Meta) | `include/c10/`, `include/torch/` | BSD-3-Clause, as exact copies from PyTorch core; `c10/util/llvmMathExtras.h` is from LLVM, Apache-2.0 WITH LLVM-exception — `llvm-LICENSE.TXT`, and `torch/headeronly/util/BFloat16-math.h` contains musl code, MIT |
| LLVM | `include/executorch/runtime/core/function_ref.h`, `array_ref.h` | Apache-2.0 WITH LLVM-exception (`function_ref.h`); NCSA (`array_ref.h`) — `llvm-LICENSE.TXT` |

Kernel registration that `tools/gen_kernels.py` generates from PyTorch's
`native_functions.yaml` and the templates in `tools/codegen/templates/`
carries the same terms as ExecuTorch itself.
NOTICE

# SBOM manifests for esp-idf-sbom. sbom.yml takes its name, version and url
# from idf_component.yml; the two sub-manifests set their own.
FLATBUFFERS_VERSION=$(sed -nE 's/^#define FLATBUFFERS_VERSION_(MAJOR|MINOR|REVISION) ([0-9]+)$/\2/p' \
  "$ET_ROOT/$FLATBUFFERS/include/flatbuffers/base.h" | paste -s -d . -)
if [[ ! "$FLATBUFFERS_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "ERROR: could not read the FlatBuffers version from $FLATBUFFERS/include/flatbuffers/base.h; found '$FLATBUFFERS_VERSION'." >&2
  exit 1
fi
cat > "$OUT_DIR/sbom.yml" <<'SBOM'
supplier: 'Organization: Meta Platforms, Inc.'
originator: 'Organization: Meta Platforms, Inc.'
license: BSD-3-Clause AND Apache-2.0 WITH LLVM-exception AND NCSA
copyright:
  - Copyright (c) Meta Platforms, Inc. and affiliates.
  - Copyright Arm Limited and/or its affiliates.
  - Copyright 2026 Mwaniki Njagi.
  - Copyright (c) 2003-2019 University of Illinois at Urbana-Champaign.
manifests:
  - path: sbom_flatbuffers.yml
    dest: priv_include/flatbuffers
  - path: sbom_c10.yml
    dest: include/c10
  - path: sbom_c10.yml
    dest: include/torch
SBOM
cat > "$OUT_DIR/sbom_flatbuffers.yml" <<SBOM
name: flatbuffers
version: $FLATBUFFERS_VERSION
description: FlatBuffers serialization library headers
repository: https://github.com/google/flatbuffers
url: https://flatbuffers.dev
cpe: cpe:2.3:a:google:flatbuffers:{}:*:*:*:*:*:*:*
purl: pkg:github/google/flatbuffers@v{}
supplier: 'Organization: Meta Platforms, Inc.'
originator: 'Organization: Google LLC'
license: Apache-2.0
copyright:
  - Copyright Google Inc.
SBOM
cat > "$OUT_DIR/sbom_c10.yml" <<'SBOM'
name: c10
description: PyTorch c10 and torch header-only utilities
repository: https://github.com/pytorch/pytorch
supplier: 'Organization: Meta Platforms, Inc.'
originator: 'Organization: Meta Platforms, Inc.'
license: BSD-3-Clause AND Apache-2.0 WITH LLVM-exception AND MIT
copyright:
  - Copyright (c) Meta Platforms, Inc. and affiliates.
  - Copyright 2005-2020 Rich Felker, et al.
SBOM

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
  echo "set(ET_ESPNN_SRCS"
  sed 's/.*/    "&"/' "$SCRATCH/espnn_srcs.txt"
  echo ")"
} > "$OUT_DIR/et_generated.cmake"

sed -n '/^menu /,/^endmenu/p' "$ESPNN_RUNTIME/Kconfig" > "$SCRATCH/espnn_kconfig"
if [[ ! -s "$SCRATCH/espnn_kconfig" ]]; then
  echo "ERROR: $ESPNN_RUNTIME/Kconfig has no menu to include." >&2
  exit 1
fi
{
cat <<'KCONFIG'
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

KCONFIG
# The ESP-NN delegate's options, as a submenu.
sed 's/^./    &/' "$SCRATCH/espnn_kconfig"
echo ""
echo "endmenu"
} > "$OUT_DIR/Kconfig"

sed -e "s/@ET_COMPONENT_VERSION@/$ET_COMPONENT_VERSION/" -e "s/@ET_COMMIT@/$ET_SHA/" \
  "$TEMPLATE_DIR/idf_component.yml.in" > "$OUT_DIR/idf_component.yml"

ET_DIRTY=$(uncommitted_changes)
{
  echo "# GENERATED by examples/espressif/build_idf_component.sh -- do not edit."
  echo "executorch_version: $ET_VERSION"
  echo "component_version: $ET_COMPONENT_VERSION"
  echo "executorch_commit: $ET_SHA${ET_DIRTY:+ (tree had uncommitted changes)}"
  echo "generator: examples/espressif/build_idf_component.sh --out $OUT_DIR_AS_GIVEN$STAMP_OPTION"
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
  echo "  - models ship with the application; the only .pte is examples/hello_executorch's"
} > "$OUT_DIR/VERSION"

echo "[5/6] Component glue and metadata written"

# ─────────────────────────────────────────────────────────
# 6. examples/hello_executorch: the idf_component_smoke app with its
#    model, exported by this checkout's export_smoke_model.py, and its
#    kernel bundle, made by this component's gen_kernels.py.
# ─────────────────────────────────────────────────────────
EXAMPLE_DIR="$OUT_DIR/examples/hello_executorch"
mkdir -p "$EXAMPLE_DIR/main"
cp "$TEMPLATE_DIR/examples/hello_executorch/CMakeLists.txt" \
   "$SCRIPT_DIR/idf_component_smoke/sdkconfig.defaults" "$EXAMPLE_DIR/"
cp "$TEMPLATE_DIR/examples/hello_executorch/main/CMakeLists.txt" \
   "$SCRIPT_DIR/idf_component_smoke/main/main.c" "$EXAMPLE_DIR/main/"
sed "s/@ET_COMPONENT_VERSION@/$ET_COMPONENT_VERSION/" \
  "$TEMPLATE_DIR/examples/hello_executorch/main/idf_component.yml.in" \
  > "$EXAMPLE_DIR/main/idf_component.yml"
"$PYTHON" "$SCRIPT_DIR/export_smoke_model.py" --output "$SCRATCH/smoke.bpte"
"$PYTHON" "$SCRIPT_DIR/idf_component_smoke/split_bundled_program.py" \
  "$SCRATCH/smoke.bpte" --atol 0 --rtol 0
cp "$SCRATCH/smoke.pte" "$SCRATCH/smoke.io" "$EXAMPLE_DIR/main/"
"$PYTHON" "$OUT_DIR/tools/gen_kernels.py" --pte "$EXAMPLE_DIR/main/smoke.pte" \
  --out "$EXAMPLE_DIR/main/smoke_kernels"

echo "[6/6] Example written"

# Sanity checks.
for f in "CMakeLists.txt" "project_include.cmake" "et_generated.cmake" \
         "Kconfig" "idf_component.yml" "VERSION" "LICENSE" "sbom.yml" "include/et_model.h" \
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
