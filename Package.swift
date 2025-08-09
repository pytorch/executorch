// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250809"
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
    "sha256": "7652999c064de1e8bde674bdacb5012c49d54462d7c07cf87f5820c51acb885b",
    "sha256" + debug_suffix: "aea7563e6dd23593de3c9632a7dbd9d850832116f0314a3434fbc0382b2b38b0",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2dff2b3d5d8475a4cb7acdf729188d9a1114619154f5822dc83a1cce3c5dd6ce",
    "sha256" + debug_suffix: "c8326bf601c3abe37c5f780f877d32d44f327b95a431040d619c9ad652228952",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "9872513242071fed20cd21f993b8402136d3ec4d82ce0d5220375a70565d3593",
    "sha256" + debug_suffix: "f6945c74041711e7eb4aba88649ee56c703037fb2e0b52a447f981f5ee9c99fd",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "5b7253ea2382e720ec8c83a5af9fb07b19e0ef34eaf964de8c1a41a88cbe2062",
    "sha256" + debug_suffix: "cf7459527c98c9aca2634c6e98f204b97010ca7cf7079792ebc64750584c5929",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e6cdbe62e2e7e8f27631d8a77c4b26344e861bc2f3c2c8e62f598148d41bab45",
    "sha256" + debug_suffix: "d8d757a99ec75b2b7c23261f37e0d8a8bdc88e5d5526b71a1422958e888234f1",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "0d2664d73f80885e0059af7aeb180f1505f393085838917a83779ccf0ce8f7ed",
    "sha256" + debug_suffix: "138716238cfe7b5a389ac26bb241a169efaa08953d49597a0d61c7129e0916a4",
  ],
  "kernels_optimized": [
    "sha256": "27f9de5382aa8788086f4334cd6efd384f54f987bb63888c858d24f41a0c25bb",
    "sha256" + debug_suffix: "5ec62c8bb7fedbe7c5f145e01b00d02e79f07c1e53f8d6ab7a0e5fa39671b5e1",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "6b087d17104733d73f8af4c79dac115dfdd457c48e663f64b9e638d734a82729",
    "sha256" + debug_suffix: "5a28f2906039bc9f9d50adb5007a9c20cb99b17dd344923142ba48b2ce6b859a",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "9b32cba4d95f30f253fa54f85e691323f6392d4989c676d5d681a973106fc82f",
    "sha256" + debug_suffix: "743b43e79f4b6ef47f3fccabd5a045b40e90d39688f4b8ce508968f5b149de6a",
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
