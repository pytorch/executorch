// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260814"
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
    "sha256": "b01b812ac950a3ac4c86e8c1fe686b1705341535343a9245b629ce561b093eea",
    "sha256" + debug_suffix: "98f65b87fdf94c640a0978b672c373d7d2d71241ac786e6e3ee472a59c758d6b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "77ccde04cc611ed0e71830d5f53670123ac0fb9b5a0c70c16f60a3204c9fbbf0",
    "sha256" + debug_suffix: "7578dd4bfcef5ed75e90a16c0165a0361ff9fa0c7d5b26289dae0ef34b1dcc61",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "527f3f1a4eac29611c5e319f3da170bbdfb4527adcf13f213421c95b09c7b46b",
    "sha256" + debug_suffix: "6a0409ccfd8fb64dbf082d7b9f5e4b368e6a2ea439fed3881a1b34033094b6c2",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "c8de6c8af7ea56c01e00ad397a327f92e5bff05e5d98d230927fb7e01dad7922",
    "sha256" + debug_suffix: "36053af13a65302a53124602a5e162b822fa2a51461f5100bbb713780f0d4e9a",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "7855fcff75c4717f0a22c964b61e4ad68f2c9524cd94a61f5a6fd59fe74a4d11",
    "sha256" + debug_suffix: "df5f5412b26c7527e52ff643a77dee30fb9edd9777fc1007c2608ba368bf78ff",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "144556e0fa563ec7749f195e5de46b8bf03c0eb855df3b39bb9d79c152e45d91",
    "sha256" + debug_suffix: "a362a2720c4fcde0f86d81ca7d2eb516638aeb16c6d4ef282e5c352f767e7f4b",
  ],
  "kernels_optimized": [
    "sha256": "a4fe798654b9b3d3c077cf83c0c4fc64732e5eb2a55bca83ab28894ae84ace5d",
    "sha256" + debug_suffix: "a8542d28ad029e61314f324180c22f355e3d6f40ef06fd24d4c799414a7942bf",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "4e1f7173ba295e5c758f9028225dd74fc8adf20a4ceb1a015899b192ae1ebc36",
    "sha256" + debug_suffix: "1585cc7fdf4e8a795d20bf85c469e184c8205830b083d2c3470a9c32d6ab6efc",
  ],
  "kernels_torchao": [
    "sha256": "f253abbce315ad4d565a202d81b82151711b1100e03ad834936dffe63cb7b1b3",
    "sha256" + debug_suffix: "76cc3deed223414d36a398e3d9ea0d2e95a27f4d26accea3d745e5d9b95634c1",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "6478722bb0a76fb24c6c235b2958154d8511e328098bfa0e0c359170919402d2",
    "sha256" + debug_suffix: "07b3404bd73e74149d27a855c8dcea294ae927a17ac799fc25ef23c0dabc9080",
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
