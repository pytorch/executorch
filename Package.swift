// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250614"
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
    "sha256": "f46547ac3c91eeef16de82a902a6b997b648455f39e91f9f5b6cdd7d92bb2f0f",
    "sha256" + debug_suffix: "8122280f2359252ab5460b7d9fa4e98ab0e3f6f4391c08e0d432b8a56fda0f59",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "100b434c2c306f4a680e3eaf61a0b60cef9c86a22bd5b1da78c44f32a228b74e",
    "sha256" + debug_suffix: "827fc918494918e6cf12f9f5108153ed9f135381685f14491f0c4afa971c7793",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b3e6a5ee2cfe16d56a30f5f94c167fe763b2eb912074de8783c2cae7ea036dfd",
    "sha256" + debug_suffix: "0822b047c39cca83f3b343baef8ab9d9b222d266c05a85e18312a1952ff79923",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0962775fe5cd434b907a4eb963facadd15c7b677ade318b80f3fdf210166d434",
    "sha256" + debug_suffix: "9116fbce6bb124e24cb81bbf66869a83c341e8755142761c9cc8b2dd55138d3a",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "6c8b79de78cd4014b3a4898264c431762890ba5465da488a77489fe7e8561150",
    "sha256" + debug_suffix: "50679d963f32f40557d6f96f2f177fb6e2af03ecc567791c987290375e5133fb",
  ],
  "kernels_optimized": [
    "sha256": "defed5acd27e72327921e7e364520eb7f202ea7792b1f38452dcbd2bcf956939",
    "sha256" + debug_suffix: "5b893d94bcda06750ee6d6b30f35cbb6be12b18072163d1be906d286a1e08371",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "c78da36d4db905c33c51d3d97589372c36c82a850b9a477feda825102c14211b",
    "sha256" + debug_suffix: "a6bdecc09f9d2497d0f640aedd4e5e179db634e9811b54495fd9d0458334241c",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "27c3dfc22b08080f4a289828ac11d6ef5ce1195bcc80238b7e5c5598f4fbc06b",
    "sha256" + debug_suffix: "a3a2329d80912e146af23d0dd61aadada99ad0beb8466d2446f38737f2af0739",
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
