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
#                                  packaging, PAL and C API): HEAD must be at
#                                  vX.Y.Z-esp-idf.N (git refs cannot hold ~),
#                                  on release/X.Y, after vX.Y.Z, and differ
#                                  from it only under examples/espressif/,
#                                  .ci/, .github/ and docs/.
#   --version-suffix -dev          X.Y.Z-dev.<HEAD's commit date in UTC>, with
#                                  HEAD on the nightly branch, whose commits
#                                  ExecuTorch's nightly wheels are built from.
#   --allow-dev-stamp              X.Y.Z-dev.<HEAD's commit date in UTC>, for
#                                  builds from any commit; never published.
#                                  Without an executorch package that matches
#                                  the component, it leaves out the example.
# All but --allow-dev-stamp also refuse uncommitted changes to what the
# generator reads, and every stamp needs ExecuTorch at the top of its own git
# checkout.
#
# Derived from examples/arduino/build_arduino_library.sh with an ESP-IDF
# output layout and the shared ESP-IDF PAL. The component vendors every
# kernel source of its kernel libraries (portable, quantized, the Cadence
# backend's generic kernels and the esp_nn:: op library) and the ESP-NN
# delegate, but registers none of them: a
# project's kernel bundle, made by the component's tools/gen_kernels.py
# from its models, selects the kernels at build time.
#
# Host requirements: a Python whose executorch package matches the component
# by tools/gen_kernels.py's rule, which is a package built from this commit
# or, for a release or respin, that release's package (to export the shipped
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
# Trees the generator copies whole, so that files git ignores in them ship.
COPIED_TREES=(runtime/core/portable_type/c10/c10 runtime/core/portable_type/c10/torch
  "$ESPNN_ROOT/include/espnn" "$CODEGEN_TEMPLATES")
# Everything else the generator reads from the tree.
OTHER_INPUTS=(version.txt LICENSE shim_et/xplat/executorch/build/build_variables.bzl
  extension/data_loader/buffer_data_loader.h extension/flat_tensor/serialize
  backends/cadence/aot/functions.yaml .ci/docker/ci_commit_pins/pytorch.txt examples/espressif
  install_requirements.py install_utils.py)

# Uncommitted changes to what the generator reads: to tracked files, or
# untracked files where it globs, including in the flatbuffers submodule.
uncommitted_changes() {
  local globbed=("${KERNEL_SRC_ROOTS[@]}" "${HEADER_ROOTS[@]}" "$ESPNN_ROOT" "$CODEGEN_TEMPLATES")
  git -C "$ET_ROOT" status --porcelain --untracked-files=no -- "${globbed[@]}" "$FLATBUFFERS" \
    "${OTHER_INPUTS[@]}"
  git -C "$ET_ROOT" ls-files --others --exclude-standard -- "${globbed[@]}"
  git -C "$ET_ROOT" ls-files --others --ignored --exclude-standard -- "${COPIED_TREES[@]}"
  if [ -e "$ET_ROOT/$FLATBUFFERS/.git" ]; then
    git -C "$ET_ROOT/$FLATBUFFERS" ls-files --others -- include/flatbuffers \
      | sed "s|^|$FLATBUFFERS/|"
  fi
}

# An ExecuTorch tree inside another repository must not take that one's history.
if ! ET_SHA=$(git -C "$ET_ROOT" rev-parse -q --verify HEAD) \
   || [ -n "$(git -C "$ET_ROOT" rev-parse --show-prefix)" ]; then
  stamp_error "the version stamp reads ExecuTorch's git history; generate from the top of ExecuTorch's own git checkout."
fi

ET_VERSION=$(cat "$ET_ROOT/version.txt")
if [[ ! "$ET_VERSION" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]; then
  stamp_error "version.txt must contain an ExecuTorch version in X.Y.Z form."
fi
DEV_SUFFIX="-dev.$(TZ=UTC git -C "$ET_ROOT" log -1 --no-show-signature --format=%cd \
  --date=format-local:%Y%m%d)"
if [ "$ALLOW_DEV_STAMP" = true ]; then
  if [ "$VERSION_SUFFIX_GIVEN" = true ]; then
    stamp_error "--allow-dev-stamp picks the suffix; drop --version-suffix."
  fi
  VERSION_SUFFIX="$DEV_SUFFIX"
fi
VERSION_SUFFIX_AS_GIVEN="$VERSION_SUFFIX"
# A release commit can carry its last RC's tag too.
HEAD_TAGS=$(git -C "$ET_ROOT" tag --points-at HEAD 2>/dev/null || true)
head_has_tag() {
  grep -qxF "$1" <<< "$HEAD_TAGS"
}
# Whether HEAD is on a branch, checked out or fetched from origin.
head_on_branch() {
  local ref
  for ref in "refs/heads/$1" "refs/remotes/origin/$1"; do
    if git -C "$ET_ROOT" rev-parse -q --verify "$ref" > /dev/null \
       && git -C "$ET_ROOT" merge-base --is-ancestor HEAD "$ref"; then
      return 0
    fi
  done
  return 1
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
    RESPIN_TAG="v$ET_VERSION-esp-idf.${VERSION_SUFFIX:1}"
    if ! head_has_tag "$RESPIN_TAG"; then
      stamp_error "a respin stamp needs HEAD at the tag $RESPIN_TAG, which names the respin since git refs cannot hold ~."
    fi
    RELEASE_BRANCH="release/${ET_VERSION%.*}"
    if ! head_on_branch "$RELEASE_BRANCH"; then
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
    CHANGED_SINCE_TAG=$(git -C "$ET_ROOT" diff --name-only "v$ET_VERSION" HEAD -- . ':(exclude)examples/espressif' \
      ':(exclude).ci' ':(exclude).github' ':(exclude)docs')
    if [ -n "$CHANGED_SINCE_TAG" ]; then
      stamp_error "a respin may change only examples/espressif/, .ci/, .github/ and docs/ since v$ET_VERSION, but HEAD also changes:
$CHANGED_SINCE_TAG"
    fi
    ;;
  -dev)
    if ! head_on_branch nightly; then
      stamp_error "a published dev stamp needs HEAD on the nightly branch, checked out or fetched from origin."
    fi
    VERSION_SUFFIX="$DEV_SUFFIX"
    ;;
  *)
    # The only other suffix is the one --allow-dev-stamp picks.
    if [ "$ALLOW_DEV_STAMP" = false ]; then
      stamp_error "--version-suffix must be empty, -rc.N, ~N or -dev, not $VERSION_SUFFIX; --allow-dev-stamp makes an unpublished development stamp."
    fi
    ;;
esac
if [ "$ALLOW_DEV_STAMP" = false ]; then
  UNCOMMITTED=$(uncommitted_changes)
  if [ -n "$UNCOMMITTED" ]; then
    stamp_error "a published stamp needs a checkout without uncommitted changes to what the generator reads, but found:
$UNCOMMITTED"
  fi
fi
ET_COMPONENT_VERSION="$ET_VERSION$VERSION_SUFFIX"
if [ "$ALLOW_DEV_STAMP" = true ]; then
  STAMP_OPTION=" --allow-dev-stamp"
elif [ "$VERSION_SUFFIX_GIVEN" = true ]; then
  STAMP_OPTION=" --version-suffix '$VERSION_SUFFIX_AS_GIVEN'"
else
  STAMP_OPTION=""
fi
# The torch that install_requirements.py installs to build ExecuTorch's wheels,
# which declare none, and the CPU index it installs it from. Read before
# --version exits, so a stamp check also catches a pin this cannot read.
TORCH_REQUIREMENT=$(sed -n 's/.*"\(torch==[^"]*\)" if use_pytorch_nightly.*/\1/p' \
  "$ET_ROOT/install_requirements.py")
if [ "$(printf '%s\n' "$TORCH_REQUIREMENT" | grep -c .)" != 1 ]; then
  stamp_error "install_requirements.py does not name one torch==X for the wheel builds."
fi
READ_TORCH_URL_BASE='import sys; sys.path.insert(0, sys.argv[1]); from install_requirements import TORCH_URL_BASE; print(TORCH_URL_BASE)'
if ! TORCH_URL_BASE=$("$PYTHON" -I -c "$READ_TORCH_URL_BASE" "$ET_ROOT" 2> /dev/null); then
  stamp_error "$PYTHON cannot import TORCH_URL_BASE from install_requirements.py: $("$PYTHON" -I -c "$READ_TORCH_URL_BASE" "$ET_ROOT" 2>&1 | tail -n 1)"
fi
TORCH_INDEX="$TORCH_URL_BASE/cpu"
if [ "$PRINT_VERSION" = true ]; then
  echo "$ET_COMPONENT_VERSION"
  exit 0
fi

if [ -z "$OUT_DIR" ]; then
  echo "ERROR: --out is required." >&2; usage; exit 1
fi

# The shipped example's model and kernel bundle are made with the executorch
# package, so it must match the component by gen_kernels.py's own rule. A
# development build without such a package leaves the example out.
EXAMPLE=true
if ! INSTALLED=$("$PYTHON" - "$TEMPLATE_DIR/tools" "$ET_COMPONENT_VERSION" "$ET_SHA" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
try:
    from gen_kernels import versions_match
    import executorch.version as v
except ImportError as e:
    print(f"no {e.name or 'executorch'} module ({e})")
    sys.exit(1)
print(f"{v.__version__} at {v.git_version}")
sys.exit(0 if versions_match(v.__version__, v.git_version or "", sys.argv[2], sys.argv[3]) else 1)
PY
); then
  if [ "$ALLOW_DEV_STAMP" = true ]; then
    echo "WARNING: this development build leaves out examples/hello_executorch, which needs the executorch package built from this checkout at $ET_SHA (reinstall after committing or pulling); found $INSTALLED." >&2
    EXAMPLE=false
  else
    echo "ERROR: the generator exports examples/hello_executorch with the executorch package, which must be built from this checkout at $ET_SHA (reinstall after committing or pulling), or for a release or respin be the $ET_VERSION release; found $INSTALLED." >&2
    exit 1
  fi
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
  \( -name '*.cpp' -o -name 'BUILD*' -o -name 'BUCK' -o -name '*.bzl' \) -delete 2>/dev/null || true

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
# The commands that install the executorch and torch this component's kernel
# bundles need. torch comes from its own index, which is not let supply
# executorch: download.pytorch.org's test and release indexes also serve
# executorch builds, as X.Y.Z+cpu, which pip prefers to PyPI's X.Y.Z.
TORCH_INSTALL="pip install \"$TORCH_REQUIREMENT\" --index-url $TORCH_INDEX"
case "$VERSION_SUFFIX_AS_GIVEN" in
  "" | "~"*)
    INSTALL="$TORCH_INSTALL
pip install \"executorch==$ET_VERSION\""
    ;;
  -rc.*)
    INSTALL="$TORCH_INSTALL
pip install \"executorch==$ET_VERSION\" --extra-index-url https://download.pytorch.org/whl/test/cpu"
    ;;
  -dev)
    INSTALL="# The nightly index keeps about 60 days of wheels, executorch's and a dated
# torch's alike, so move to a newer dev version before these age out.
$TORCH_INSTALL
pip install \"executorch==$ET_VERSION.dev${DEV_SUFFIX#-dev.}\" --extra-index-url https://download.pytorch.org/whl/nightly/cpu"
    ;;
  *)
    INSTALL="# executorch built from $ET_SHA, in an ExecuTorch checkout of that commit:
./install_executorch.sh"
    ;;
esac
# Split at the placeholder rather than substituted, since bash 5.2 gives & a
# meaning in a substitution's replacement.
README=$(sed "s/@ET_COMPONENT_VERSION@/$ET_COMPONENT_VERSION/" "$TEMPLATE_DIR/README.md")
printf '%s%s%s\n' "${README%%@ET_INSTALL@*}" "$INSTALL" "${README#*@ET_INSTALL@}" \
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
| PyTorch c10 (Meta) | `include/c10/`, `include/torch/` | BSD-3-Clause, as exact copies from PyTorch core apart from the generated stub `torch/headeronly/macros/cmake_macros.h`; `c10/util/llvmMathExtras.h` is from LLVM, Apache-2.0 WITH LLVM-exception — `llvm-LICENSE.TXT`, and `torch/headeronly/util/BFloat16-math.h` contains musl code, MIT |
| LLVM | `include/executorch/runtime/core/function_ref.h`, `array_ref.h` | Apache-2.0 WITH LLVM-exception (`function_ref.h`); NCSA (`array_ref.h`) — `llvm-LICENSE.TXT` |

Kernel registration that `tools/gen_kernels.py` generates from PyTorch's
`native_functions.yaml` and the templates in `tools/codegen/templates/`
carries the same terms as ExecuTorch itself.
NOTICE

# SBOM manifests for esp-idf-sbom. sbom.yml takes its name, version and url
# from idf_component.yml; the sub-manifests set their own.
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
  - path: sbom_torch.yml
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
# c10 and torch/headeronly are exact copies of PyTorch at the commit
# .ci/scripts/check_c10_sync.sh checks them against, apart from the generated
# cmake_macros.h stub.
PYTORCH_COMMIT=$(cat "$ET_ROOT/.ci/docker/ci_commit_pins/pytorch.txt")
if [[ ! "$PYTORCH_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: .ci/docker/ci_commit_pins/pytorch.txt is not a commit; found '$PYTORCH_COMMIT'." >&2
  exit 1
fi
cat > "$OUT_DIR/sbom_c10.yml" <<SBOM
name: c10
version: $PYTORCH_COMMIT
description: PyTorch c10 header-only utilities
repository: https://github.com/pytorch/pytorch
url: https://pytorch.org
purl: pkg:github/pytorch/pytorch@{}
supplier: 'Organization: Meta Platforms, Inc.'
originator: 'Organization: Meta Platforms, Inc.'
license: BSD-3-Clause AND Apache-2.0 WITH LLVM-exception
copyright:
  - Copyright (c) Meta Platforms, Inc. and affiliates.
SBOM
cat > "$OUT_DIR/sbom_torch.yml" <<SBOM
name: torch-headeronly
version: $PYTORCH_COMMIT
description: PyTorch torch/headeronly utilities
repository: https://github.com/pytorch/pytorch
url: https://pytorch.org
purl: pkg:github/pytorch/pytorch@{}
supplier: 'Organization: Meta Platforms, Inc.'
originator: 'Organization: Meta Platforms, Inc.'
license: BSD-3-Clause AND MIT
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
  -e "s/ *# @lint-ignore$//" \
  "$TEMPLATE_DIR/idf_component.yml.in" > "$OUT_DIR/idf_component.yml"

ET_DIRTY=$(uncommitted_changes)
{
  echo "# GENERATED by examples/espressif/build_idf_component.sh -- do not edit."
  echo "executorch_version: $ET_VERSION"
  echo "component_version: $ET_COMPONENT_VERSION"
  echo "executorch_commit: $ET_SHA${ET_DIRTY:+ (tree had uncommitted changes)}"
  echo "torch_requirement: $TORCH_REQUIREMENT"
  echo "torch_index: $TORCH_INDEX"
  echo "generator: examples/espressif/build_idf_component.sh --out DIR$STAMP_OPTION"
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
if [ "$EXAMPLE" = true ]; then
  EXAMPLE_DIR="$OUT_DIR/examples/hello_executorch"
  mkdir -p "$EXAMPLE_DIR/main"
  cp "$TEMPLATE_DIR/examples/hello_executorch/CMakeLists.txt" \
     "$TEMPLATE_DIR/examples/hello_executorch/README.md" \
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
else
  echo "[6/6] Example left out"
fi

# Sanity checks.
for f in "CMakeLists.txt" "project_include.cmake" "et_generated.cmake" \
         "Kconfig" "idf_component.yml" "VERSION" "LICENSE" "include/et_model.h" \
         "sbom.yml" "sbom_flatbuffers.yml" "sbom_c10.yml" "sbom_torch.yml" \
         "THIRD_PARTY_LICENSES/README.md" "THIRD_PARTY_LICENSES/flatbuffers-LICENSE.txt" \
         "THIRD_PARTY_LICENSES/llvm-LICENSE.TXT" \
         "include/executorch/runtime/executor/program.h" \
         "include/executorch/schema/program_generated.h" \
         "src/et_model.cpp" "src/platform/esp_pal.cpp" \
         "tools/gen_kernels.py" "tools/kernel_libraries.json"; do
  if [ ! -f "$OUT_DIR/$f" ]; then
    echo "ERROR: generated component is missing $f" >&2; exit 1
  fi
done
if [ "$EXAMPLE" = true ]; then
  for f in README.md main/main.c main/smoke.pte main/smoke.io main/smoke_kernels/kernels.json; do
    if [ ! -f "$EXAMPLE_DIR/$f" ]; then
      echo "ERROR: generated component is missing examples/hello_executorch/$f" >&2; exit 1
    fi
  done
fi
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
