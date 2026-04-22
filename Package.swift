// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260422"
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
    "sha256": "4e48513efd8a47034cf0022d58b2f3c097ac42a31331c017356c0d8caee9a4f6",
    "sha256" + debug_suffix: "6acdc576ef0f173cbc645c09bfe9caa16bd4bd812c40f5de512900b6bb754e02",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a0a00dd09c0f76c75b4749c56a34cd2d617d66fb18e6db6d9cfdf4cf9be20a7a",
    "sha256" + debug_suffix: "49deeb0662266a802641651ce1f304c51fb352a8b0ffc872395ad770ffc259aa",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "5b72df3ed31fe72c3b39d0dac97772bfdcd73c98dcc6a43d12109685d1ea8357",
    "sha256" + debug_suffix: "2b3694ddbf17843266658780b0083daf203f965ef1bfc56c7c810f0c21a4b578",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "a4863807c224e4b06bf057345817541029e6dc781cb17c2c907ac95c840909b5",
    "sha256" + debug_suffix: "6eb7d008ab1d2b844b954b36acd8b4025ca3a96be48c33448991b6d223fd64c9",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e7919e947ea3a68f0f511d621f399eecb0553da5d85d4f18a1f607c3d836a4b2",
    "sha256" + debug_suffix: "699e4273bd01a0a0a163959b70cda0efb3dd40ff0028cf6ac300d27770370335",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "8f20f32b200275e6ddccfa845bd2b339cfcffb6aaf1701b55c433d4465b0ed24",
    "sha256" + debug_suffix: "b634d7aa06932138a47578536c38781b96d862d2ec69d56d7ad22f1fadb98f49",
  ],
  "kernels_optimized": [
    "sha256": "52413542fa9e867e1e1c3720050fcf902667b1c89116f3f62e700cc197b6f13d",
    "sha256" + debug_suffix: "d8d35c183e039812020f87a7e068d5868b75f6569040e5108e2c2a992b615e83",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "79897df116e5d3409ccd4beaa22e0ba8a65aa79e4a4c76f2a39971ee59a3c772",
    "sha256" + debug_suffix: "3b3d3808bed09a495ab37dab4aa5e007a6765f39551f49ac14f9dded7b1b370a",
  ],
  "kernels_torchao": [
    "sha256": "ba3b52140668c6af0ccf46c809219d4be4123b8fb61f80cd5a7e21878aecfb75",
    "sha256" + debug_suffix: "342e31fe8d8cbb925bcff8435daf2cc39009b7d3ded92ab19e0a336fdf09849d",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "639fae3ccf93876ff32f1c0782aaee94bfd71745efb4d279ac580e91a9e93cad",
    "sha256" + debug_suffix: "8ab9a697fce8b09d9b799bf91db0942ec66c11b6ad643a37cc2ae3c021725796",
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
