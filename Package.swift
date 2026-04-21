// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260421"
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
    "sha256": "30561f79bf8430794930ffc090ba6871fe7e4fd3ec74b52199cb7ae824010b67",
    "sha256" + debug_suffix: "b863c303db948e175ce6b120d848e2a4631063423e756546202b36e24347ee54",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a28b53d4a66689336f3f4fe24b9de5ea839d00c8e29ce5e59a80a9dafa0580b3",
    "sha256" + debug_suffix: "317684c26813a7c42d0d8587563e9f1e8b0f156810025ff4646db64eec34c116",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3b85ce713f4a45ac7be613e64d304e693c7701323d75288aec48452b8697aa9a",
    "sha256" + debug_suffix: "d4ed1c2d5c23df279d44d1b535e2f97fd075fc1e5031fda1d654d89b82343270",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "226a691105aa0605094a28626d3723bbdd2590044a599c8741e5f962db8bf376",
    "sha256" + debug_suffix: "feea7f9cde3153c65558da30b9308d00b24c0d6e9888780adad260de7011d2c1",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "a0ea9e4f0d1364b59629803b005fd0259cfc733b10d34f17c9d08bdfa8db9aa9",
    "sha256" + debug_suffix: "2b7f49300a89316f6c3040f9bbf5141ed4504d9123afd39247741d1ebd1911ca",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c9d75c1c62ab0869fba492dfeb269f98ebb32a9d1089a448e35663aba657bd89",
    "sha256" + debug_suffix: "dabb8f53b28a57a8531785fc0496870ed6e8e75f3ed5367ec74cdbd82c678c0d",
  ],
  "kernels_optimized": [
    "sha256": "7a7f6ad5fa1459f294705a328bb4f1ce56ef409d1d1dba7b9e4bdf1b6711b01f",
    "sha256" + debug_suffix: "1776e0dda4465b277f9fd106c4c8a94cbc0ca7cdacca80ac4ebc6dcde0cc0c98",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "4a9e3b9c4207da531299a0ebd2d91ef3075c839fbac1e6fd1484df57e0f4dffb",
    "sha256" + debug_suffix: "d9f493faed2719afd3e5f42c2c35dedf246c8d9dd24f1581f2c7df177d1141ab",
  ],
  "kernels_torchao": [
    "sha256": "dbe64f6442c945260f0a5b0a56a34ac4d468ca4d26d534d3e150d77a2c283f44",
    "sha256" + debug_suffix: "e989f2c616760a3d5b5195be2d7df9368e5862f027f6a9b6357fa9f8eb5c833b",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "838b89ddfe58dedd44c94aed39e5110754013381c4a311c9cb7afa305e1ec33e",
    "sha256" + debug_suffix: "4d4f12a2b3854aef1af0791c4531b68bf66d7717101fb241540288e7ff284880",
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
