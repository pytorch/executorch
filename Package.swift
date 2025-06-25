// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250625"
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
    "sha256": "671eba9ffdbd04c3e8886c176f9859f957b8ca6c7ea92b356d410bb48a71c30f",
    "sha256" + debug_suffix: "e1a783f0e300c5f8e7357ac7db28c9b31ac196865d898da8121e1c2ae387ad7f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "610a7d04e50fc4211ad0775ce6cdebd42bb4f8f8a6f1e7d5ea3281cd79e86e0a",
    "sha256" + debug_suffix: "f36c16b09bda6692da11d3921af3cb6f5bb6ebecb6a5b86e737a32dc660e70fd",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "859cf529452f8a2a0faebdeb18d24810789e65febea008c63b80166269a5d984",
    "sha256" + debug_suffix: "93dee7fad38b6edc322c0f465ec06018b9144ddeffeeb32b7e6ceb2723c5f1d5",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f3756ca8d4dd441b4216b72e92ec798ff6f0dfae2c84e00dfa6c76ad5ad72128",
    "sha256" + debug_suffix: "378d1014a44723952e22444d701b3a2fdb14079eeb7889913a781782ed94bd42",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "4d4e7e3a6f98807ae41275c7fb1cc66b50524e50365db0ebde38ac888549f75d",
    "sha256" + debug_suffix: "2ad97377c567d80120403f721a7d909b441190c316a8304eaf8a147b26e5fcdf",
  ],
  "kernels_optimized": [
    "sha256": "8ce30d087d7d6ab4a62724eaab40cb55c6a4409b1dd767e976952ea9a9d9b3d9",
    "sha256" + debug_suffix: "385c38568510e1dc19a0b94298fc277aef51486c2253e822b12c3f7462edd969",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "6a6d45a54ed5a8e514f5e63c91b22517fa05ac34d92dd9c4c9759382065b22ee",
    "sha256" + debug_suffix: "a1b886bafafad8f7b97ba626a8117b18ff1bdf669daf5fceba338ea95aa92098",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "b4be46e012d74a9990250ccb1143401cb55d660d7e785309f0aa4101d44e6378",
    "sha256" + debug_suffix: "3aeefc3095b3f39fb0237ff0b5085e4f6d9d4a1a43d73ca78f4dd1cf29da02a8",
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
