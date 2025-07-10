// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250710"
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
    "sha256": "a457d27dad3827cf191e64e888caece02cb2517a0613af62cd9da4b455043436",
    "sha256" + debug_suffix: "61ced1001966c04f5c2c43160da0831f7d44d55ecf44f86ecc6ee84e4886a565",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "bbbd5a48d959dbdbc1177645f007e926dd0ad66d9e697ee08b87625895ca2671",
    "sha256" + debug_suffix: "433ae7be0818f555ed924f522bfe695c9dfdca8fc46ba5c556dc0b1b15028a4c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "2866a155a0dc0efaf3920cb9c71725072b86fff77e01a72e5f06e1d5705a3557",
    "sha256" + debug_suffix: "85ef473662c5ed82f1a39bbd95fa025c8b65cf84c742c3ed426d552cdc0d9120",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "2908cf5dc7c2bbb1570ffedbe672cc8cfdbd46d0f7c074623934bfcf86043add",
    "sha256" + debug_suffix: "8d7cfeffa734772c335bb7153b3315459b12e434e2911d6ffb724e4bfe0c6b2f",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "959816caef1b4383b261083264c05966a7999e9814c7ca15c741814cd0eaf470",
    "sha256" + debug_suffix: "630035c28da5787fb5d07f6e11da31891dbe5f56b43ee7af9a899578415fe763",
  ],
  "kernels_optimized": [
    "sha256": "a2e0acd39a591f28e2d3e872883398a8a04b03fa809d0946e63a69a7b765aa4a",
    "sha256" + debug_suffix: "a7c3d38fbb6bb2960af18a1c96f7591e7d7468775c0dfb7eee3ca993fa07b7d6",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "7a368d9d8651633d6d4a9a5e33d883a209803b3427cb41edce3fc40c3c9e3fa8",
    "sha256" + debug_suffix: "5a55798543e9f94d9e8a692c65b11e5f4c2647da9607ecea0358f09f064bcf8a",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "c96465b488e1ad8379e4901e7b8b1e6a547b0108010810ae78d27c65ba35d2f0",
    "sha256" + debug_suffix: "064eda48df25046015d75503218e2e37895c4112d2e88dad7797775e9f37e7bc",
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
