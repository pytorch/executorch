// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250612"
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
    "sha256": "ff87cc482858863241377fe985719626852afdf2acd1c7f9d9db08bdbbcbbff6",
    "sha256" + debug_suffix: "b577716c87bf0ec49c9bfe9f4518c3916a21703e7c7de5fc43d4441499fa0df9",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b65508ccd52fd552492491ce8d6148e0dc81286635dfb7bd41c56d4d0a163896",
    "sha256" + debug_suffix: "0a5022e85ffff8d93e5298f0afd933710262816d73b33d8a05b5c78d866ecfe1",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "2695c1b0772d4d47b3e3f1e8c6368f4e558bfe042f0dce9c60918a21ce67af15",
    "sha256" + debug_suffix: "3e0d61db165090933dd73c0b5c7a4657fb7ea9672bbfa54c837cca3664221afc",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "726a5525519f6ad30b9980cbdc04caa10f0771a50e776c8f69de61080a3fee4d",
    "sha256" + debug_suffix: "6738baca6bc23fcd76ca99eb6260d0e7d6d173e45dd6fe4c226c913bac36e392",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "5978ea1e6ff07ba940ee38844014bcb3e5eda81cbbbfc5645221e8767045b3f5",
    "sha256" + debug_suffix: "018a1548647e49358620431f777918d0b47ce810fb0823fa3ab164394df6f7ac",
  ],
  "kernels_optimized": [
    "sha256": "9dd4fd30119da576a12f5120109d666a730311131b66897dd8e8afbeea29fb5e",
    "sha256" + debug_suffix: "b7b51b976914683f9833e3d4fe10c3b0e2ef8d9b49266ba821e6026950c45b7c",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "6be9716f024d3bf66914f727efda65129653a813e19f8a5e756b401faefb186c",
    "sha256" + debug_suffix: "36c06212a699ca018e3e648c3457344cb79d27854b626f32ec24f1e6be7f5b1c",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "13656f596d197fae2e4547ac39552cc9591035e2f24d2c2f459c89038203e560",
    "sha256" + debug_suffix: "a5712fc7a52d401616e0d32f27a6e4269cf1dedc18454aa7ee9592b7e933bcf8",
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
