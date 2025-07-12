// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250712"
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
    "sha256": "b75271e86e2d6b380e6e36b25053663d084a98eccedc6adbcc078e33a414f530",
    "sha256" + debug_suffix: "a81771e19011865ac85bded91fcaa8922f5afb30a4254b7ace9339db2e3e4dd9",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "13ce50b6c2763e54cf020f8dbca1b91ca8a0b35f6bf8b8216f34e9a0a046c2e7",
    "sha256" + debug_suffix: "af41f71710f753f6eb0101d480113392da1afbb66e9d51155ec4ae535e5f2a5f",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c3d0368d6eb8242f48e5ad5f8dcbca6b4e62a93ec06658e8cc78ee748f213e87",
    "sha256" + debug_suffix: "c18c3facbb626d298a06e5f4e3cdd6e98aee7f32b878e572fcb12bc6e90cb3b4",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "b634a4df9f19b18033a942bf391c707a4a526949b286516bcb7d4271370c2a52",
    "sha256" + debug_suffix: "b9e893ccea965b34dd3b70edad064d20a2c9bc9ab42ba12487b64edb3c9341f8",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_llm": [
    "sha256": "cf0658cdb4938bf4582161f9534d1e1513ec9bbcb9a111ea088fd827b224c8da",
    "sha256" + debug_suffix: "f22aa0d866c8a5da13a17919bac50cfbd164265cc4d4426713aeb20a5efde177",
  ],
  "kernels_optimized": [
    "sha256": "642a358d1744062de07b3651b445f9d91a6d8c55248f063eeede78b4be01c0c7",
    "sha256" + debug_suffix: "e5c9988e4381209bb5449d9980bd94ddde0e803e7ca73a2321ea19bc6fed8921",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "13d3f885a7235d5c8d2fb8f109126f6639c29b29d67f49829952756aded646ef",
    "sha256" + debug_suffix: "c9488fc9bec3d26a1ecfb0f6567eca6f72cbb0089f562516e41d6009087f5546",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "add3e3e0bc839426b5805ad0c2507e581ffd415d6c87eb89692848971e5cfe1c",
    "sha256" + debug_suffix: "00b291d58ebe89dab1251398368268431edf7bf02b3146627043dae63592414e",
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
