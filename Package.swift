// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251103"
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
    "sha256": "f71a0db4da4b11ebbb0e9218c098b144ebeef9cbcaed30f659e3eb3f0c31258a",
    "sha256" + debug_suffix: "3016aa69fe1709b22f95ef495867f0827d6e8afdb9896392aab9f433d7ab6537",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b478e3bd809b45e3f2efcb47ba48a08f118ff9b7de27b1caebf4b5cdcaf5035e",
    "sha256" + debug_suffix: "e80609792b13ca3000a0bbc5f12ab4af53327f82ee1b05d48f580617f61a5a67",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "69912e23a877a6b5808458a4bd2641677676baf8ecb7d4d4ab5fe56563344b52",
    "sha256" + debug_suffix: "6cd5061d011bfd222989caca90a719860a41a18caf085db522f36dbab91124be",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f601153640ad777f5083531f3442f9bbaf5a8199d0cf2cd9518966a984b1d906",
    "sha256" + debug_suffix: "8b3a9c6ae2af6086bcc76df42bf6a32068c121d37a6f7ada3325da8a0efdc4ab",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "bcb4253360c47225f2e30c82628825dc5e42ead712bda9aa6fe2bb5333d4cb96",
    "sha256" + debug_suffix: "3a906ace95e0d455a22e07ff77a0175cfc5f5b3dee19f5d7cdb8fd1ce997b42f",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "3d1986d404f8dadc5659e9dc936b66d0f9f6b0f9018ecb572ee3207a4772afff",
    "sha256" + debug_suffix: "fc0b55846cdbd3287b585283dba3d5dec0dbfd7840b57ee682a11104a67776ec",
  ],
  "kernels_optimized": [
    "sha256": "2f8271f2c6d22afee7545fb99b5e6bc7881e555cc8ab71c1c9f86623b7c73bbe",
    "sha256" + debug_suffix: "b6d929c1e6e44dfb3bd0d4b1cc03ae56b1723fb22b32968db9878d4188e1edca",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b320df8aad19c0982703e3a76edcfdcae7eba352406b4049f05f188c6709c7cb",
    "sha256" + debug_suffix: "aa364277140f86affe2019e48701fc1699939f23fe662be478dccc4f24c72b39",
  ],
  "kernels_torchao": [
    "sha256": "e51c24a235279d0829649ff572c5c574696fed05cda4974ef61030ba02337ba9",
    "sha256" + debug_suffix: "8107b8354aa4d2583b12f6ee8041e794adb817170d39132a9c05e94bbe534ea6",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "8db82dff978f53dd51d01ecf91016cef62cf69dce7a9d64410a3ca4aa36bbd45",
    "sha256" + debug_suffix: "4d83c953d9257016d9bec39798f7d1c7b379767bafe3e56af46c33f4d9066491",
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
