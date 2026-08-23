// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260823"
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
    "sha256": "4196ef59883ab0719f38e501bcab97558da8f6e3ae26fba0cb923fd84d5e3749",
    "sha256" + debug_suffix: "5779caaefebde7c9d1f4384c15dd5a6a2af311765c4e6654a48a4b99409f5b0f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1133788a4c8a9a13e0f2ed2dc7357b2aa237e037ba8b75675828b4bb212c73e9",
    "sha256" + debug_suffix: "ef84bffa4e0991618960163cbb37abe2eb1a6e40f6da03281e0cf25ef921e47e",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d5844509d4a60bca706b8df68984c5f7b3627fcfcceda5e5d3a0d288c89bab50",
    "sha256" + debug_suffix: "9d7342184573489787ea94457645a2750866e7c876b204bbc8143437618d5e3b",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "079130ac3317c0578fb2ec43a181651ccf1823aa6f919fe5c9e12a5fd87e62b0",
    "sha256" + debug_suffix: "1a1a1a3c8a703ff00a6ca25c44b1a6eea2f116e6cafc34582f5877385030a561",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "8f35dcdb4a65b29d42ab6e1165bcf1069b387912ff61b7dbfb675446b37c0ef4",
    "sha256" + debug_suffix: "d0f0c5f39d81d64366d0a3156a1a4de0be87fce1decc93de3148e2eac1a8f480",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "4d5b3f6a7a4d586dcf25d04e2ca30aa69b6f541f125d3199c495c31765779f37",
    "sha256" + debug_suffix: "1b1c62bbeb6d9f3b9532cd6dde1a58bb5c457d7c56b94aa4f3b8ce7fdb94efd7",
  ],
  "kernels_optimized": [
    "sha256": "a741d18d566474f39dc6647a56d6605801401c8dd9a8019ec30323abfa73676a",
    "sha256" + debug_suffix: "ab043efeacc5ffed11841e2d9bd14e36a2b47c9b6ce3f5bf297dc6a55807834a",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9e2b9d7cc98b4b3562867659fcf1233d99134f8e73f3c5b0bdd44d5c17e10ba0",
    "sha256" + debug_suffix: "c8e13ea411913195b0e948bf4722344d43660cfb02e16756113cc25344a2d593",
  ],
  "kernels_torchao": [
    "sha256": "a41cc86aafa5d9959a7c36de973a47fb7cdc7d4fb46f12c69a02d0e6ec5eb534",
    "sha256" + debug_suffix: "c016528d0a5851f5074d2891db8fe8b16bb1815afec06a4523ac3adf78b383b5",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "cccfd529351e01550732e14f4774a3ac8cdcd9a0cee013c2e2f59442bdaff203",
    "sha256" + debug_suffix: "36ae6483dc02e8c3d2f3d81567881a9da0368a5b7ff02bacf0f1fcc46fde35a9",
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
