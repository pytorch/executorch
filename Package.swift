// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250808"
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
    "sha256": "e81c5fb62ced5e8cf9f264c4f846bd1897f93bb8717cf7017da7ac4dcd453885",
    "sha256" + debug_suffix: "61214ccecc191b766f27e003b16c55865531750f4d480a05384b90206883a50a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "e738af043673c38a915efe0ce47f5ade204d1974ca8420f44be774db366a09fc",
    "sha256" + debug_suffix: "41a17048f9ec0b44e609ed6a8418c5e87b57b09f23353bad29792fb11d5ae520",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ac4d5c7a41a1239037ce36475f53e64683829a08b2d310c8bbfae3d26256681a",
    "sha256" + debug_suffix: "6f48258335820fa57422d9eb97e837a4d315c3b680ad8ddfc1516e0f78def1af",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "3c3fe78457971d3aa3fb8d1db61b214a8c5c6347ec1303037bb9a8c2f5695c6c",
    "sha256" + debug_suffix: "c8875cf3332bf75373baca02c504700d679b229cf499bfbab76934efbe81494f",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "56dda9a4be7c6b706b7910886562180202c3d755cd87a89d8eda08970b894825",
    "sha256" + debug_suffix: "3fc1b8be1cd3826d830b76bc5848c5f0a578f7ea70a2d22596a082cb176c896b",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c70afa6f5858e6283232c10bd933bbd41c21a4cd1bc79f93dac62b82e7a0894f",
    "sha256" + debug_suffix: "83c99b78a7ba8d08c1c69209b44d0dcadbb9ac336acc7438b56209cc8c213eff",
  ],
  "kernels_optimized": [
    "sha256": "95472645774fde311904614b917a0e8ad008e675cbc7333bffb962f339ce40a7",
    "sha256" + debug_suffix: "d2b033a9f0da0d0a1db53c698a1ff6b82c15223cca918360dd1b2f6e3e4356cd",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "0c36821f445eaa0c138c3e0f5cc553053e435bbdf507dca053399f2c8e2bba80",
    "sha256" + debug_suffix: "da97d218ad79ebc1fac723cb4cbf0509e8f0445b6d9288e7d39eacb2fd0a7a63",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "91c27df95125d92b576bef7595b9cdaa8114fe1ea264b7088e7ea38e26a15438",
    "sha256" + debug_suffix: "67af0d5acfe113e7f3916476dc354a834fc9d4d82857e341010530e07e27e614",
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
