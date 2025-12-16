// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251216"
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
    "sha256": "4562a22e0117b3e80e26d8b83d3437e24106d08a01f9addc889f641640c96eb0",
    "sha256" + debug_suffix: "333ae5ecee5533fb9dc7bd428a1b61b5c1e058c21405d509fd7d8570095d87bb",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "bd0bbae16393f9f8b19c0be05ae5f3bbc137baa9977b0d2bd5dcc75190fb32f4",
    "sha256" + debug_suffix: "8fec7bf9c98fa5a4e2eb1048ef6a9c2c4398258ce32b0dc85137481406310e46",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a7de46477708c9d75c83a216ffc016c675eeb40f5cbe09597c483f64ebe40ae6",
    "sha256" + debug_suffix: "6850f4c4b4131348bc70239792f03cac203bb1b1c8702da3dda6f00f7b651027",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "c925b142b0069e5433f01af3690f5e2ecbece41e511614310b1f9ca2f6b3ce5c",
    "sha256" + debug_suffix: "9615c8f06fe086156c010f0cdd472cff6f44ff731addacb974788501a86ef77c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "ea4c459c1cc72bebc8489e066c4e4776838bf673fa28ea232dd4ae0e28f3ce9b",
    "sha256" + debug_suffix: "b054afb90162bf04c6e2d8f73a366e605b03e07d9d7ce6680bd79f90eec14aec",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "2d6f6e8781f87606e514b0a33ee7870b323ff711f089829fe7dcadd4fca4a7b4",
    "sha256" + debug_suffix: "741fc7f93a0ca71aabb9335be1207e0e03cc364bed85bfef2f54e247bbedbd4f",
  ],
  "kernels_optimized": [
    "sha256": "02512201cc249ad13e53943c95249ae8b128c79660999fed6ff5b6d5f4cfc560",
    "sha256" + debug_suffix: "6a15c91b5dcce71596a6548cb8b1912609b35ee8dec7b752ad443d8fddd65fb9",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "c20749c3f407d25f57e87295a99505904f65aceb8cd676fc344af36022b45b87",
    "sha256" + debug_suffix: "9afccff44c38be2b041b28fefb431501120a19d0e3e1a7d78c11759cb8c97aba",
  ],
  "kernels_torchao": [
    "sha256": "c75699b871119ac44e33f16210e8e2fbb1d3496cceaa3b1d79f7b06fb12c1e75",
    "sha256" + debug_suffix: "042237e30b049b7b8378eb22b772d9d4cfa2bca1fe51e101035def8358074b19",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "06865cd9fdf6415242c9e41085c2694293d6954b764e828517721ada875c4069",
    "sha256" + debug_suffix: "fc12a087dd0b911a4169af6dbfcd7cc792b8140e44be9526e97f9f7b58c5216a",
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
