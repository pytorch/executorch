// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260820"
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
    "sha256": "876ed9333dd4590e7d5cd28a622f02a8682082576a2396094fb19b50e5ca239d",
    "sha256" + debug_suffix: "d2d90d85557feb19aa930bca2d0de9908ae36e017574a3b1fe17e8f79b8b2a0f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a7a42f603cfd324159eb130e8560e443dea403faa5d9b0cd7a09765a54ca438e",
    "sha256" + debug_suffix: "a05830715aedb878f21eba377c1d8f8778af6ceff5f6b591412040e92b989518",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a33c41cb762b9edc47f8d92e3b0314eccfc0b2d9a7714e5f50deb32c8c3f4f21",
    "sha256" + debug_suffix: "c2ad8e91274dd035664585e3a636f48e4ec9b200ce81546feee13ec8094a3c48",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "a3bd2d8584263b4c9ce2ef20272986df7d10f9ebee17a9845709ebe150cd19ad",
    "sha256" + debug_suffix: "806bdd91a98c54f8afb31f9efd36a3eebf9c359ae74c354111234e1986978a31",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "8fd9ad1dcc65ac66b7261096c63e275307f3caedbd1de5cdda293dc56d5c630e",
    "sha256" + debug_suffix: "0c432eddb2f169175d5a27f332eeabf3f145d651767cd3904396859f08c0a660",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "8d94695d3a498fd3a61ad55799f643742e8f5a848dd7a9d2161c6d3594c637a5",
    "sha256" + debug_suffix: "eb4527779cbe7053e88d4d4d91f86853a865736cef2d91cebafe4f397fff7586",
  ],
  "kernels_optimized": [
    "sha256": "5f735812073eb5abd86a3b2300acd1fce71f5e5dec0fb6870b771a7239987305",
    "sha256" + debug_suffix: "67780d4dcc19256233a66b3b8d7b7a363b5617f544c0864500d15b1577c55def",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "90d8e564f16f24c73d62902edce61c9bbe92244ace806010c2b0f4382cbdf6ae",
    "sha256" + debug_suffix: "dd5abe46d25ee726f3f502b5b2b483de7e6ea921443d3d22c7daaeb69cfa97af",
  ],
  "kernels_torchao": [
    "sha256": "6b6b508c4c8998cb60c9df4449cdf0f2329b35321bab102e16a3d206d3a6335f",
    "sha256" + debug_suffix: "3ff7f4671856a2521f4407fadb1cb5aeb2e690637901b095792246121dba45b4",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "eb8de3e24092210a15c45588c716bcad64d258241f679bfcdfc94e4a74486162",
    "sha256" + debug_suffix: "4c84e9e502cce1d45a0acf61a23c9a40163ff8534225dc52762d96ab486aeaed",
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
