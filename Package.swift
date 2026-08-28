// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260828"
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
    "sha256": "cce9d6f806fa8f158754341328a238df6e0b1f99b34c4ef754778299fea58848",
    "sha256" + debug_suffix: "c0efce3dcbed28eee79751783a38d346f66e7faad49c9fd2f6637386ac339c25",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ea50ea1b2e9c253a7470c149e8f00c6481de156049ac5019bf3219860d9c31d3",
    "sha256" + debug_suffix: "06d860affb2b3130e193539d35841ff2c052dd3e92f4c7672eb5af525041ad23",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "40c9c08c34d52e62dce5ad1a8a9de37f66de1c67d59d434496a3c88ad546d53a",
    "sha256" + debug_suffix: "22345c44cf9df67c5e65695f142dbdacc6dcf328f2f57dba08b9ce4831d5fd5f",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "d3d77d28aa0a7b17bfcd91d5a8340c3714ca3fd1e0b8e2fed0051e81bb550ec0",
    "sha256" + debug_suffix: "65ec2da9d17aebcd40362ba95d7d5b708850da3cdc78c73c550df5eedc6bca06",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "deef338c5a10a5965b8d201be6f26a99e24fd1d2fbe68a919c3dcbed1f2c2972",
    "sha256" + debug_suffix: "a697b52baf66806e88e04b502728b51847e77c8581fd85abd5f85d47fe0e81f2",
  ],
  "kernels_optimized": [
    "sha256": "5cf7c4e8df04b53dae773c7ae386043c7d8d923efbe28f1c2cbafc845d85d735",
    "sha256" + debug_suffix: "577b4cc1fee74f0aab72403416064db0a27826c91328195f4967cc95b12c5737",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "c633b52dda164fda069f42026bdcce9b45aaa8bad491c7553e9333709c98b6ac",
    "sha256" + debug_suffix: "64c27a4541a9066b027d351b4f3f5ac4ae57bbcf54d4f025c62b7a99180f70da",
  ],
  "kernels_torchao": [
    "sha256": "80e66a6064943a3e688a72f7e5ce5bc4a36052d8047ef36eb9181c63a2cb86ec",
    "sha256" + debug_suffix: "73ac0b3d2bd46d03d2485e116e9b09b5c94cfebfa31d4031b6184fa5d1db6b84",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "6fae8ec043a04f898f4c8bb8154a0059c5918ac4ccf6fc4dfdca9588ddcc9ddc",
    "sha256" + debug_suffix: "fbc58bb06347535237b356024e6ff4b1fe969c7eceb9c08d0c1582b4ea00560e",
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
