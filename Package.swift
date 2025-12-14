// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251214"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug_suffix = "_debug"
let dependencies_suffix = "_with_dependencies"

func deliverables(_ dict: [String: [String: Any]]) -> [String: [String: Any]] {
  dict
    .reduce(into: [String: [String: Any]]()) { result, pair in
      let (key, value) = pair
      result[key] = value
      result[key + debug_suffix] = value
    }
    .reduce(into: [String: [String: Any]]()) { result, pair in
      let (key, value) = pair
      var newValue = value
      if key.hasSuffix(debug_suffix) {
        for (k, v) in value where k.hasSuffix(debug_suffix) {
          let trimmed = String(k.dropLast(debug_suffix.count))
          newValue[trimmed] = v
        }
      }
      result[key] = newValue.filter { !$0.key.hasSuffix(debug_suffix) }
    }
}

let products = deliverables([
  "backend_coreml": [
    "sha256": "bf164d5aa1440ac6bed4aaf5019e531d6487c6e7afd876e3401fbd007b246e9c",
    "sha256" + debug_suffix: "28592537725c5300e1d86f8386a8fb8096249b71c1f1a9760a50375a2ef456fe",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "f034a3a61a849f4d49599d902ac2a42b0f1bdf63040c462501d5acc48dfc3d36",
    "sha256" + debug_suffix: "b2b0cddfac583f99e87106859ee1a702b1b023bb7f158567c345b8ee17132580",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "cca96715b690208fc4592987b2c637bf603f0f57139725622122ee97e2682020",
    "sha256" + debug_suffix: "7b7d19e7e3eefd1a035c51f9c9d8043f8ca361b1395385e7d431c5abae18cf7d",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "6a3aaf188d9ebbe78425e4ed52eb84c9e912e489ec760c3d53656ed727f349c4",
    "sha256" + debug_suffix: "349b70666360a4e448477ee09236b34cd179dd0eacef59fd4313de9410e936ef",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "83d2737470337201a72dbfeb1b081ac7b0471ab9b3403538207b02e021040b1a",
    "sha256" + debug_suffix: "e5bd938940df21377234403f2d7716e5c4804f7cd9226c4ae57571e17d57b446",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "e56da38e6db0b0698621551df15c48f266bd012ca7fec308f5bb44b849704448",
    "sha256" + debug_suffix: "9c192d87fff0dde52d99b67c295a81eec70771f125990af64399a085c073f5dd",
  ],
  "kernels_optimized": [
    "sha256": "e2a0ea5c4ad2c71c1c3946a06461f16b2bd5572550fe07af561ee0d725b1b45c",
    "sha256" + debug_suffix: "79089b9951b0362c52576c041b824cab3be097aa423ed3c491d5a8f31c17029d",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ecfdbfce6f9c15b5553457b4a7858f377595c2a787cd4cd96be363e7a03f516b",
    "sha256" + debug_suffix: "e2269fd890dce602f59d9fe720ebf2745594e7627310db5eaf1e2b97e88321ad",
  ],
  "kernels_torchao": [
    "sha256": "883706424601dfc123e2efdab6908a0f0c4aa06cb4ff98d8b5142af6090da37e",
    "sha256" + debug_suffix: "ec73d77b3b0c973e125843ce2ea3514c86c2b61161c02696155c1c21e0e286d7",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "9d8189a23ffdeacba576911e6082d4fd31ab4c69256af4ef3ba666556760d323",
    "sha256" + debug_suffix: "a52beae3a07607d1e26ce222bc39a7619edca1ec3e8da8af80620f9582ec7e9b",
  ],
])

let packageProducts: [Product] = products.keys.map { key -> Product in
  .library(name: key, targets: ["\(key)\(dependencies_suffix)"])
}.sorted { $0.name < $1.name }

var packageTargets: [Target] = []

for (key, value) in targets {
  packageTargets.append(.binaryTarget(
    name: key,
    url: "\(url)\(key)-\(version).zip",
    checksum: value["sha256"] as? String ?? ""
  ))
}

for (key, value) in products {
  packageTargets.append(.binaryTarget(
    name: key,
    url: "\(url)\(key)-\(version).zip",
    checksum: value["sha256"] as? String ?? ""
  ))
  let target: Target = .target(
    name: "\(key)\(dependencies_suffix)",
    dependencies: ([key] + (value["targets"] as? [String] ?? []).map {
      key.hasSuffix(debug_suffix) ? $0 + debug_suffix : $0
    }).map { .target(name: $0) },
    path: ".Package.swift/\(key)",
    linkerSettings:
      (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
      (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
  )
  packageTargets.append(target)
}

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v12),
  ],
  products: packageProducts,
  targets: packageTargets
)
