// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260310"
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
    "sha256": "c7f3a7069dd150393790cd7c729c8e31ac444a2dc277c82bc48081106b145dde",
    "sha256" + debug_suffix: "0a0f821e556f177bde7ac4eaeef8a384ea44a407f49edcfb6a9753fcaf747d48",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "9790ce584c53fb6e6fd8460037fd1a80231024d22e1ca21c282dec92ac96b43c",
    "sha256" + debug_suffix: "cbf4c56a84814b3f46875c5c8d608efbc387ab1bd95e5c00da0a54d2a57834e5",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ea75e6ab06e55bd7632df5747929580c1758dccec7b83c67d69f97450ab0accb",
    "sha256" + debug_suffix: "a4e85857f5c8ef01bcaaa0ad1b9b3399ac4af09a100fc430823599c1cda751bd",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "5134fbce8bdf7abfa62b2e21c6d75c9d043507b67a7302a32e47d76d4f8b7ee2",
    "sha256" + debug_suffix: "73ba65f3eb0ffecfaa79abb8f54e7e35eab276a8623d19a9b23489f06305a2db",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "0c9d4d5a698f575906419ba9a13d7dd6279e3afc807ce0a14693b1126aaf25d2",
    "sha256" + debug_suffix: "a31093859e8f5e937a498cf3a1685b6700e64509ea8d8d3026e62b3cb4e8a720",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "948dea4ae9a53004099e828e3bdc82d94c4215ffd33eebd441d5855662cad1d2",
    "sha256" + debug_suffix: "2ddd93ef5c4bea606db10ae1bb095d872aa0b1f1ea7dbbb5607eff37ff7a6753",
  ],
  "kernels_optimized": [
    "sha256": "8b43eb7300827dd8a98aa33a363e34aa67b299a93b707caa9df0f6ff5bc85ca0",
    "sha256" + debug_suffix: "a92847d9ebdf366a927eb2de6e7ad23ea628824b88348b0a48239db4049a8d88",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "c26fdabb0f75b4058ffbe0848e7fff6e470b3b120a30fd282e141d50fddb61bf",
    "sha256" + debug_suffix: "c2b18be554b15661fd1ec8442bb3cca13bb765977ab6ced55c5aafa58f64ce60",
  ],
  "kernels_torchao": [
    "sha256": "2bca1f50d9766d233ca41219a5429ded655ca243622652b9775e90c2c670616e",
    "sha256" + debug_suffix: "ddac2a145c633a496b0eb4bbfec326b51e92f3e847181227b16b8d568b402ef8",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "5e830e7634fab328fcea641cf59a1ee4ad8248f5bc7cbf4bb2c631fc0b9af59d",
    "sha256" + debug_suffix: "3f098895e99d5d68a26abfa48a2f8b5218caf794d57799a1d634483024768335",
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
