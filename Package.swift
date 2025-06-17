// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250617"
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
    "sha256": "99e6a2f5f55f8d700eb45044eee1361a6900df00f6a4abbf6b367f9cd32cb588",
    "sha256" + debug_suffix: "a5611833e6142d64eb3e5f7a7560f1047afeb688b27069fa3d73b8524b52b417",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1d380631d12bb0e8df113b7dbfe1b65a65e1df57903e670af105c681365c50b4",
    "sha256" + debug_suffix: "27f4bf98e05a02ee6b0e58e4e9ea5b8e9bffe7bc0b72247119da4b926a6ae890",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3325b308e27e9c762501c11936462eb114972b9731979c13569f9dea0e2d31f0",
    "sha256" + debug_suffix: "4dd05919df0d25f91d21cdc0eb96a982441b53d51ded25c3a0ed2b5f228d20ca",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "10ffcbfe3a5fb6aa07138fdda4f42f031893a9a423d5d4d331d6c524d832c4b4",
    "sha256" + debug_suffix: "c13617ef06b26fb2ef94d8b9b8a1aa27ba143ccede52dc4f9f4abd77f98bfb7d",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "4ca016fd359074beddb8883831b8d10992a1fb89697acfcfd87d237b7e79cf84",
    "sha256" + debug_suffix: "8303e5de9750cf40846dc43d25b17254265f6f252b2ebc5f30c46af7be6ebe85",
  ],
  "kernels_optimized": [
    "sha256": "e89651395f0d59fac844983d8d1781b1aedb8dfdbba4c6acd0d4084b69213d66",
    "sha256" + debug_suffix: "0eff237adbefc731cb8139e4e1e88eb387f11a59d04be67bc4ec7f38317f59b3",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "427df0b07a549259455f0a7e5b68bef3046203197d9e32482488e834cefa47fe",
    "sha256" + debug_suffix: "6dd72b38a17c6d08e3ddac1ae5562b2cea9de7a7ef362e57cd5eeea5a38eb37b",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e5fb87e76fec092f40a45eb5295aef5be5e5f68dbd576fc5b370e1174896e650",
    "sha256" + debug_suffix: "baab9d72e0536da5f2385404d14ebda11fbff1936d9cafad7f23ce720d4e1d88",
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
