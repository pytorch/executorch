// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250716"
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
    "sha256": "727a0963c77d6fdc72ed0aa8ce9f9f8b978b077560354e9c31da9f3207a4562d",
    "sha256" + debug_suffix: "a6f890bb0e02985196fa4c2464b4697aa6fc9307856f4fdd3c92e53b9e286c0f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "990f5ca77efede130ed5356138d952a4f17e428ba79698a01bf3f1271103d242",
    "sha256" + debug_suffix: "06568ab8513b43ab6280416e2379a7c59aef962610f7dad7fedb59dd5a46aba0",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3ac21172dee4a6cbcb16360c853170c0f683b89e9753d4ad45d0e11031b8cba3",
    "sha256" + debug_suffix: "0a31055e981332604be41b21af8d9cb370e12c6f441a2236900658d5cc34d7a7",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "96d5ce29c937649a5f41a2ed8875e89b119cd22686dc3e371208bf940fe1a5ee",
    "sha256" + debug_suffix: "59aba21d4802b0caf339e41c294bd6cf23ee5ab4e7d84494fdfb3f668587c598",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_llm": [
    "sha256": "f19431f7b22cab6d87420c95e140726ab99ca48f02359bfd0f5a510495878748",
    "sha256" + debug_suffix: "9c02f07da0dc92c44c6e3065621730b9b46a3e5d677ae85e0ca1b8609208b624",
  ],
  "kernels_optimized": [
    "sha256": "8584698738c9e2c3bb9d5299ae40d3239699ebe623b9f26a246cde4c59734dae",
    "sha256" + debug_suffix: "00862375e6a105b7d7d69c1d031ad7ae30b3602e1ce582535996199a2b290ea2",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ba9f5c8c9b8a88f115c653d2764be6d3243196bfb1757f1cb371ef1f3e4af1ef",
    "sha256" + debug_suffix: "1857a58d6c150068e83b00008a9dd0cc281f2cdbbfc97ef6df9577939d8f54e2",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a13574766e234b574568a60b7b1ef61d21f4a78f5b10160ca2e8850e129236c1",
    "sha256" + debug_suffix: "b9f256299c37967ff71f44e16c52e040aad9a83f28ddf88d8f968a553c23615f",
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
