#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATOR="${SCRIPT_DIR}/../cmake/ValidateGemma4WasmNames.cmake"
VALIDATION_ERROR="must be"

validate_names() {
  local export_name="$1"
  local output_name="$2"
  cmake \
    -DGEMMA4_VALIDATE_WASM_NAMES=ON \
    "-DGEMMA4_SPEC_WASM_EXPORT_NAME:STRING=${export_name}" \
    "-DGEMMA4_SPEC_WASM_OUTPUT_NAME:STRING=${output_name}" \
    -P "${VALIDATOR}"
}

expect_invalid() {
  local export_name="$1"
  local output_name="$2"
  local expected="$3"
  local output
  if output="$(validate_names "${export_name}" "${output_name}" 2>&1)"; then
    echo "ERROR: invalid Gemma 4 WASM name pair was accepted" >&2
    return 1
  fi
  case "${output}" in
    *"${expected}"*"${VALIDATION_ERROR}"*) ;;
    *)
      printf 'ERROR: unexpected CMake validation failure:\n%s\n' "${output}" >&2
      return 1
      ;;
  esac
}

validate_name_matrix() {
  validate_names 'create$Gemma4Mtp_1' '1gemma4_mtp-profile.1'

  expect_invalid '' gemma4_mtp GEMMA4_SPEC_WASM_EXPORT_NAME
  expect_invalid 'bad;name' gemma4_mtp GEMMA4_SPEC_WASM_EXPORT_NAME
  expect_invalid 'bad name' gemma4_mtp GEMMA4_SPEC_WASM_EXPORT_NAME
  expect_invalid 'bad/name' gemma4_mtp GEMMA4_SPEC_WASM_EXPORT_NAME
  expect_invalid 'bad\name' gemma4_mtp GEMMA4_SPEC_WASM_EXPORT_NAME
  expect_invalid . gemma4_mtp GEMMA4_SPEC_WASM_EXPORT_NAME
  expect_invalid .. gemma4_mtp GEMMA4_SPEC_WASM_EXPORT_NAME
  expect_invalid .hidden gemma4_mtp GEMMA4_SPEC_WASM_EXPORT_NAME
  expect_invalid 1factory gemma4_mtp GEMMA4_SPEC_WASM_EXPORT_NAME
  expect_invalid bad-name gemma4_mtp GEMMA4_SPEC_WASM_EXPORT_NAME

  expect_invalid createGemma4Mtp '' GEMMA4_SPEC_WASM_OUTPUT_NAME
  expect_invalid createGemma4Mtp 'bad;name' GEMMA4_SPEC_WASM_OUTPUT_NAME
  expect_invalid createGemma4Mtp 'bad name' GEMMA4_SPEC_WASM_OUTPUT_NAME
  expect_invalid createGemma4Mtp 'bad/name' GEMMA4_SPEC_WASM_OUTPUT_NAME
  expect_invalid createGemma4Mtp 'bad\name' GEMMA4_SPEC_WASM_OUTPUT_NAME
  expect_invalid createGemma4Mtp . GEMMA4_SPEC_WASM_OUTPUT_NAME
  expect_invalid createGemma4Mtp .. GEMMA4_SPEC_WASM_OUTPUT_NAME
  expect_invalid createGemma4Mtp .hidden GEMMA4_SPEC_WASM_OUTPUT_NAME

  echo "Gemma 4 WASM name validation passed"
}

verify_product() {
  local javascript="$1"
  local expected_factory="$2"
  local expected_output_stem="$3"
  node - "${javascript}" "${expected_factory}" "${expected_output_stem}" <<'NODE'
const fs = require("fs");
const vm = require("vm");

const [javascriptPath, expectedFactory, expectedOutputStem] = process.argv.slice(2);
const knownFactories = [
  "createWebGPULlama",
  "createGemma4Mtp",
  "createGemma4MtpProfile",
];

async function main() {
  const context = vm.createContext({});
  for (const commonJsName of ["module", "exports", "require"]) {
    if (vm.runInContext(`typeof ${commonJsName}`, context) !== "undefined") {
      throw new Error(`fresh VM unexpectedly defines ${commonJsName}`);
    }
  }
  vm.runInContext(fs.readFileSync(javascriptPath, "utf8"), context, {
    filename: javascriptPath,
  });
  for (const factory of knownFactories) {
    const type = vm.runInContext(`typeof ${factory}`, context);
    if (factory === expectedFactory) {
      if (type !== "function") {
        throw new Error(`expected factory ${factory} is not callable`);
      }
    } else if (type !== "undefined") {
      throw new Error(`unexpected Gemma factory published: ${factory}`);
    }
  }

  const requests = [];
  const sentinel = new Error("stop before WASM fetch");
  let rejected = false;
  try {
    await context[expectedFactory]({
      locateFile(path) {
        requests.push(path);
        throw sentinel;
      },
    });
  } catch (_error) {
    rejected = true;
  }
  if (!rejected) {
    throw new Error("modularized factory resolved before the locateFile sentinel");
  }
  if (requests.length !== 1) {
    throw new Error(`expected one WASM request, observed ${requests.length}`);
  }
  const expectedWasm = `${expectedOutputStem}.wasm`;
  if (requests[0] !== expectedWasm) {
    throw new Error(`expected WASM request ${expectedWasm}, observed ${requests[0]}`);
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
NODE
}

case "${1:-}" in
  --validate-names)
    if [[ "$#" -ne 1 ]]; then
      echo "usage: $0 --validate-names" >&2
      exit 2
    fi
    validate_name_matrix
    ;;
  --verify-product)
    if [[ "$#" -ne 4 ]]; then
      echo "usage: $0 --verify-product JS EXPECTED_FACTORY EXPECTED_OUTPUT_STEM" >&2
      exit 2
    fi
    verify_product "$2" "$3" "$4"
    ;;
  *)
    echo "usage: $0 --validate-names | --verify-product JS EXPECTED_FACTORY EXPECTED_OUTPUT_STEM" >&2
    exit 2
    ;;
esac
