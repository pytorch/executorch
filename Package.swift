// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251223"
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
    "sha256": "726923aa85039d424dec4710e79d1d34c8638502f252560a244231aadb070d09",
    "sha256" + debug_suffix: "ac649e194b2873c2b469c0a606267a3650440b449b11c03c43499e29a7c9d5fb",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1a94122a442df5798141bc7748cfae73d4fec08c01c3c34c95052aefb57752f9",
    "sha256" + debug_suffix: "51394391ece180576177f908b71f261d83518035965e495a0d9809d46c9adcaa",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "69a76b9b6debc8edd01e4fd4d94ef5c7ae776cbe5b511bb971ded62039651a97",
    "sha256" + debug_suffix: "8f23c1ff27718201c71b960ad2c038ecb33f1232587f57778af8a70c99489b4a",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d0cc9c19e7c913029c4258522254c407383c2565a08baec9942ea2bd4fea0e76",
    "sha256" + debug_suffix: "bae37ded965d1c1907cab31fb22897158bdfaf6528b7b5fce2599d89c6d2d00f",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "580b79dbad57ea932d49014ac6c2389c3e65d9a37e312c4f5075e95392a79beb",
    "sha256" + debug_suffix: "82a1028372a9f2dca2058ff8cda20810549ad9ce8220746e15ae0ba033dff746",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "b138ebdce18ab3a3085dd5c2a5799fd3d10095f4909996bd0261fa5828f9ed9c",
    "sha256" + debug_suffix: "f2e4b7898e32449b22cb5e5c76df3a1d6fc3bc8bd8ab33b1e06464bbf0e23304",
  ],
  "kernels_optimized": [
    "sha256": "8f170025b9a3fa533336c9cc450e3c4ae84e5d73355cff5848e688d0c7534590",
    "sha256" + debug_suffix: "38ebb1db0e3536b9cb33acf821be51aed1265327ab9a95fcf094df2f28767d3b",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "693ee8ad26d54c8f3b4f19df1386d82b0f4d2eb4b947d5f20ecec681cf4fc93d",
    "sha256" + debug_suffix: "6778dfdff74cac0d44d61949115ae5b8135174be8c4492f4df4b153e6a57229d",
  ],
  "kernels_torchao": [
    "sha256": "32731b73c3afe2888a3ce03a3ee6246c1e4cfc10f5a4e0f0c666856002ef4721",
    "sha256" + debug_suffix: "6769b9e1a461f45c9e3345f6328ba602201f733ba13b00a59cb867b910f8aa21",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "880c2c2a5fe00548df2f96a0eea9389517f30bb92a68e5a1f55396759686f3a9",
    "sha256" + debug_suffix: "04689a63040c5ff5106f2e584451cd26dbf5801541ba4cb0b51f646bedfeb3bc",
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
