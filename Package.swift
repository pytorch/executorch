// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260717"
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
    "sha256": "a376623af0a007e90e368fbd7df913602467c2e59ed5489a86f182096a4642a3",
    "sha256" + debug_suffix: "e9ef52a84cc6b708b4a67330cb1842dd0e9f7ef0f30553326298d9f1d1b39025",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a864e89fa8b45976c6e0a7a4a97482ecc5b8a6b2f21e2c18178b38f18c7f5de5",
    "sha256" + debug_suffix: "47f50a45f8f9847e495390612cc72937691d3098c11ba3d3f038be3fe5fec917",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "11872cdf2d5bcd80d58970ca8474a0b2f8b5f8c77c6bf996261e15bbe5fa7fba",
    "sha256" + debug_suffix: "c702ad55d4b54ebe55d1ce0e1abb916dae3cc917e06fcf94e7defc3bf0dfc361",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "35eb97dcab0c4a291420ca15ab1d5cebb42f26d7c72134b85cede6a9648a646e",
    "sha256" + debug_suffix: "0a2f49a7ab4d4ede001dda213a71af1964fa2598fce9e0fecd2580fe65f9e68e",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "8f74f54e423a82a811d0b674d0bed900c52358b4e748d0d57a315f937655a744",
    "sha256" + debug_suffix: "75aa092412bec3cd753dbd1679237638dabd98391110f2881be25127a175062d",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "5b95e0c1633958e58a4cf1c65db94c436e3f082defc0885cf166ae7063fc1980",
    "sha256" + debug_suffix: "262e002f282d3e48ea4d29a44ed74fa44be3a4e18027b45a5d50fef68c3fc768",
  ],
  "kernels_optimized": [
    "sha256": "3fd786b16ac3cba143d51d8d3febfa28907b7a255fa284573be1a33843e6a3a9",
    "sha256" + debug_suffix: "3051bde02bee76a92c96586c9dc5a655b30eb5b310756680fe33042609f4f635",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "0aaf1d254dc318fff46b5d01628cb41691166d730a25ab1fbc34a8c274fd41d7",
    "sha256" + debug_suffix: "5eb1d1d9d62d3e516502f4d70593075d09b082e261047766679a8c1c365ed4ba",
  ],
  "kernels_torchao": [
    "sha256": "bb137345260ad7d867cc3d94971f59727da0033668f250c4f3bfe61c3aec0d80",
    "sha256" + debug_suffix: "0bf34987ebbb492e48975c554f29a2821c5f4302e63156fcd09fa4d7ba8249c0",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e05eeaa94041e0023f1ee4cb66ae4251c0fb0dbe8f923d65455cb3a22201a452",
    "sha256" + debug_suffix: "19056b5b2b5aa6b7e84580fec1f3fae7db0a3a5c104f26ce497d5916ee1af729",
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
