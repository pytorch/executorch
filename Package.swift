// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250601"
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
    "sha256": "09beb8dfeb38720400b2a34d9f2a876d1a63a50d44b2f2622c1caeaeb23d105b",
    "sha256" + debug_suffix: "6e890fb99a7e6c2c38d19aa26b46f4e5252a9890a77a7266305b6678d599cf04",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "549b19a50875600bf95aaf1682e51908f26bb76a7379e2184592b3441366f5bf",
    "sha256" + debug_suffix: "a293b5f3ac677de89b440995c17b1c947cff95744a3b28bd7470193be39d0103",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "5a664e978b281fdec3558feef1f771fdbfc26e3763ef7f3e82ecd1d9f00c3e73",
    "sha256" + debug_suffix: "dbe7b3ad03041f933ce4a686465505b59ea80d3c145bb726188ee1c11ed1d903",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e30a013964da401404bd535f8e7c61168fec41608567142e146394718a32d3ac",
    "sha256" + debug_suffix: "9757c4b106cdf30b986902484ccce177977c588cc744f1127df986cdc34e7d16",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "bd4698f5d778c0c40c624865645be14d4f0c0dd4abe5efc6c83e070a18c443f3",
    "sha256" + debug_suffix: "17ce194d776a0b729f6abc4e61b7d6eae93dd0b59e066ded3db5693f59d57127",
  ],
  "kernels_optimized": [
    "sha256": "c212a69d3a064a33c95c4bdbf78a6b820fce39e627ea69852cdc39f70f472de2",
    "sha256" + debug_suffix: "cbc2d23634c90f27909c3d40b0f14411c0cefc4373bfa641eb3aceacd037d9ad",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "d74a236a6b427a4675eda6ed312b3a1336f342924144e1d72eef75f845f85d23",
    "sha256" + debug_suffix: "0fa9886f2e3eb5665a949bdbd08d6c3f5dbcee055b89491e7416fe8831261eb2",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e2a33ca245943cfbcd60721e7e3d49076ca9ff0dbafa5fcef1ef9289be1579f5",
    "sha256" + debug_suffix: "1c2221492c1faef3db25ade7c1ff418ec2b3129360c7d9e18899d70eccf007ff",
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
