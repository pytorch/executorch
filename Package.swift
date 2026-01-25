// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260125"
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
    "sha256": "4ecd8eef3f7a08ec225f594745b65c041cd2f4153d76106501d562deed766c63",
    "sha256" + debug_suffix: "6dc99d9bb9e044ec3036fd05afb62e291c474ba2ca8b46764921192d2b4be06c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5123b2b417650a741ab189ca83542b9ccf3a26c9d4bd81ce036d9816ba6e43a8",
    "sha256" + debug_suffix: "7046d0f70be08198529ab21602512d981b8c7c6c4397a265cdf77cc1213f71dd",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "167ae751e63398e7e0071a0030d6300881569508074696cc34dc88ea5b4f7a27",
    "sha256" + debug_suffix: "7c22259cdf46446f059c302f8f618bcdcd630ad205bb65de64eb9cb8b563144c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "46ea5a619b86f03191268425cc769b7d3656b6de620ea74edc06c24934746fa0",
    "sha256" + debug_suffix: "2ef4b3195a527eda844d0965f9854eb92f01137252792e5f1e9f6bcef5cf2ff8",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "5cd6252e743ba0b8ea744e7da783bc68821096ea89c7660bb8aebbd84c5ddb6f",
    "sha256" + debug_suffix: "4043a6a831fad3d629a81f76409b3055e1f2d72b192d08592fcefd5cf3055ded",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "22f3251d380040e037b2f6d5139102b170fdfb81a3a9d8f259dbab74ce01e0d8",
    "sha256" + debug_suffix: "fc83da9f2b2d27893475e7cd130fc4fa4aac5283a7b4cc5a264c389b56b9e137",
  ],
  "kernels_optimized": [
    "sha256": "a0e3f672b63ae113962eb5d50967cecac946ef088a0e3c457fea1130bc4d99e1",
    "sha256" + debug_suffix: "15c09e67c393bcb8a91a1bbfe6439dd6be05687002be098b9f6f4f1ff865a8a9",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "8352f394c83e7e44c89fd955faf5017a15108332551dbcd85255597552b06840",
    "sha256" + debug_suffix: "a426ef5a12dfe2dac114328a2ceb9dc6a461c03c28f505a2661de68e0cb06a40",
  ],
  "kernels_torchao": [
    "sha256": "1118b5243ddcbf345c930b2ae5b20b8ecd6b4f8912f1e83d4ab28e105befeaa1",
    "sha256" + debug_suffix: "fa4bb2858a8e5bb1acd7f181b524f442116f5bca4b578c4d9c125cef13e37938",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "0a36f0d94e787c569171f2b88def4a70c49a27cc0e319e355492bfa1d8553f6e",
    "sha256" + debug_suffix: "39ebf2f41ba439a06c5541797475412a3d7d02e636be98d89fd5577d74671a45",
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
