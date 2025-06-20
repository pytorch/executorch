// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250620"
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
    "sha256": "d1d59301a7d83c343d3350732575d0ebd4f691a66feb27a91251db781c8bf925",
    "sha256" + debug_suffix: "4409e5fea180a2eaf03bb3b1feb8c25e50c3bc9b492ce6d5a0ded6333619d810",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "09bd7d8a58e25526ece0b70a3e661039da7ff5d06d8eada7e48764020c221db4",
    "sha256" + debug_suffix: "8ef7ba61ccf58eb1b8b6fb05c56df1698a00886080ac6e773df05d00b4405a29",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c41003ff5835759b8f771d6a164e081cc145809a948d70322776217e95918fc8",
    "sha256" + debug_suffix: "d61e83686ac42157d31614d3f01fe74bc592cee1081a13a7d6d21a4c2ba59bc4",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "8a1203f1f0cca982c4e51192c84fa1d72f15effe5c757f80ceee9e7da30a9162",
    "sha256" + debug_suffix: "8da2106e6e10eb1c01e2e936800ab664027aecfa8f690a78b8c03e107287dd7c",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "9b097ad8bc607f7d9ea166598826cb9efef380c16a20fc48913dd31efed248e4",
    "sha256" + debug_suffix: "afa4240047d2cbd757ee8805ae5b2bbfa2b0723f3732c3568d4c5fb8a27ba8d7",
  ],
  "kernels_optimized": [
    "sha256": "2c2e39bb3ced63d6625ee1cc29c68857d9f18b88d6b14ce982d2adfa39a3ba11",
    "sha256" + debug_suffix: "5da8f2f5b8ab0d7dfae3553ad0a1dd76b1064be89345691913492bcd379b4793",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "31e21156c5be3ae889e4ca0578fbad9dd83dcfd6e5bdb049bef839ea31ab4667",
    "sha256" + debug_suffix: "27afe066dcdffbfabe6c7b2ecf3302aab7716938779833e49d118998d1ed82ff",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "895b04bd6ed954c6d18b052e3da3004a56e7b3e7d8957bacc19a043143288456",
    "sha256" + debug_suffix: "2a57b97c20b14a9657d13114ccf10f8f89619a4edca4355ac5d2342e0d1cc81c",
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
