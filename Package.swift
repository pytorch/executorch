// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250801"
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
    "sha256": "2f5a6eff339d9b577785f1789b8acbf1b95454fc98f6b6b99616f75c2a35a7ba",
    "sha256" + debug_suffix: "7f4d403acbf91d13b60086220cf9da9fa744cd1d62857ae529dbda537a26b2eb",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7a9529b84c63f1a35127db1cf9b238fd918ff30fc24b11d26cff568e0c2e1561",
    "sha256" + debug_suffix: "0c3c24e58aaaed6dbafdb6aa35ff85e235f0d05934b518c4e83e9fc17eb6bc7c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d280229b447afac2673cfa2cf65d0a331ba1e921ebe7611d045da98018f7e6ec",
    "sha256" + debug_suffix: "09a48d37828db3220f9dca3c6f0c3e050810b2a71acd303745d0e9c48e2e86c4",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "33eef056b8bc0f534e4432948e54774078b277bacc166299f3b7e3784b56f419",
    "sha256" + debug_suffix: "91d8402d77f06298f556f6bbe622410e445f49747110bd5ff64abdbca663e93f",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "00ee469fa6b03c1551c5e2962e0290969a8a0101da114e484981a5ec17f971b6",
    "sha256" + debug_suffix: "903f179f60f2727360886ce2b59ec802667332a394f213103301229650eb374a",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "849994733789c679a090377f327e6d7236bd74dda0901df5a77e6847640a471c",
    "sha256" + debug_suffix: "756d5836f9df2b357b81fa38debecdfd0e3b6893d40fddd8730d860a8b9e7777",
  ],
  "kernels_optimized": [
    "sha256": "69e60ffe88524e0ce52de4326aa0be694c308cd592dfb2068262b46af7115f61",
    "sha256" + debug_suffix: "908e723097669f1a41afe956fb2d16dbac88a948d3f51089eb57b543c5781dbb",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "10c495d0b4ded558ebbfcdd129a321a28d946028b7d24ce3209910465ee4a2bf",
    "sha256" + debug_suffix: "29aefdfe764f4cb5ed5d317964fb589d0363bccf127235b49e99df34b1d9fff5",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "13f90198a7d83503ce87f77b14c1dd8a063939335bc10a835efbc18827160a1f",
    "sha256" + debug_suffix: "0cadeb175230da840b8aafe4317df055cdaf0948fe8db7243943f7038fc73249",
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
