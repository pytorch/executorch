// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260327"
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
    "sha256": "0fc2120c9545a92bd71e363189a6d154483d541f57f4824124f0262b42a507cc",
    "sha256" + debug_suffix: "ce20a615134aa2d27fc2b30390c94bd54e60fafbffbe6323c69b8aae29e99d72",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "f4349eb822af276f7122ba0fecbc775dfe21356d835259be7b67f610f91966fd",
    "sha256" + debug_suffix: "f22ae72d462132c26ff1554ee5d13c867488992c006dc8e39347a54cdf3e0373",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a6010e8209b85469a99e1e2142281e41e34653398e896f1bfc00d6c8a89525fb",
    "sha256" + debug_suffix: "9f321f4d0c1e1131c228fa7e688bf02893087d5d52742c07635dd52dbcfbb2cf",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "722e6f00c83c312268449843f77d0922a8272e6e602051aa69dd30701d470096",
    "sha256" + debug_suffix: "a08eb3677dc1ff940c1e547ca1cd285e3dcdc1c608d04a49c7d1d29791d5d218",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "5443c8a0a8222f59dafe37ba32376e5ca6be138e0bbf9d5d9e5086c616c85a20",
    "sha256" + debug_suffix: "34e67a5771bacbf49ff2658917a37e2c4fff51d861782b3d3d34ee28bb4e1ca8",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "4072c1d0073e12e635977090aa1b9505422355bcfd5f7c73894eb7c9d65bc052",
    "sha256" + debug_suffix: "39f4f9b279fe555127733398bcd2dd809fd5ad25f2e00fde299c00079ad6973e",
  ],
  "kernels_optimized": [
    "sha256": "bb5f448697d54b866f6a2fd09676b19752a6943e1811175af8fd3fc1d6587ed1",
    "sha256" + debug_suffix: "077d6b35beea3e23e5b4d863862a936215a76c03ebab0bda92b419a7e1b498d9",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "7ec6aa00a7df196ebebce603dab4613a8a277c3b1c80cc8b0684ae4f409b3ae3",
    "sha256" + debug_suffix: "8acd026f52caa26d5e05e07352a905d3f5986e89d24ec102f6a2e99c20d8a07a",
  ],
  "kernels_torchao": [
    "sha256": "5da94e7dadde828b5e97ce19fd01f5a12fcf0a37ac607ff99d4df7a71cefa03d",
    "sha256" + debug_suffix: "1d0b5f98518d97e451c19d8098d797c8f4e0bc11b8dc97001130fa2bb26dd817",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "0f42bc0502857eced30bc0e16a53098d679c9927dcdcd787c0f0f7572ea14d10",
    "sha256" + debug_suffix: "ea4ef3658181318e94bc1b6d16c8eef9d107912aa6c369e2fc81fdd12dcc3701",
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
