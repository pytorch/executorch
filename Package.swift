// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260325"
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
    "sha256": "84995d78df9c6e203e38a62c0e3b61e377c40a6bdf740cd691bb06374d94f6e2",
    "sha256" + debug_suffix: "0ae3f0eda74ec270309e4ab7c31f2e516856d838eff294a61bba45446ae58c8c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "59c1084399c19ea42418b930b4984ba74608ced3a1900a95f9dce83b48c9e3f5",
    "sha256" + debug_suffix: "aac874886c9c9e022b2ed7c85b796024b7476e1ae14196ec97b9260a5d01c5a5",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "4470e10f1f956638b45986730ca32cc3e5b63bfdba19e93fc5605a1e425ef289",
    "sha256" + debug_suffix: "bb9096cc3b0ea80cafdfc1efc9fe58be6443a3400f6ab84278f4093496c6e24c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e67aa23af7d7e9d85764da70e4c7f0521f55a97d3752f5292d7771a9b5c5323a",
    "sha256" + debug_suffix: "8609d004af2a6afd7997735132344d9f55e588d08b35a8216b5c4860da520fdd",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "b1c2b5badce92f7aac76e8bd6d1c1e1979bd51ffbb71e31b7a986c96c6d54a4d",
    "sha256" + debug_suffix: "5e35ebfdde896c6c95fc1198215bb064524691893265e8f0ee7164b912ed3f65",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "d885275389ad6162af58b1c50d7890048b31885a23233fb0530e2fd390e3243e",
    "sha256" + debug_suffix: "f2e84dd4c7780a4f94415b8f4200fdfba3ce901ea931437723aaed1678e3c6e7",
  ],
  "kernels_optimized": [
    "sha256": "bf57aa8d7679d58786282ddbc3d4d4e46ac8de9b02487eafb62a2550a34243e9",
    "sha256" + debug_suffix: "a5bf46d669f3946bba9fb699cbd129a83d807025bc92b3cf43770d5ba1760bb3",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "dac8d0f519efab4c961c0c98e659860d594af72b0ee40fdec65d165b3086354c",
    "sha256" + debug_suffix: "3428e28cf95d81ccc7ecfa7ce3c0a24682dd9bae35373c22b2c3709f9a75b4cf",
  ],
  "kernels_torchao": [
    "sha256": "a10be20b61fe41ab7fd39b0cbca4d9599f747b89cbc8e731e04ce921099f75a0",
    "sha256" + debug_suffix: "84a1c533ad49e9ecf6cdd2ebc08d9a1a7fc76fdd556e15dae9b5f94a8bbfcd2f",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a8ff706d0bea32f4b9d9ba970a3783e6d0effb95e8b96aaa36b3685ac3572fff",
    "sha256" + debug_suffix: "1cb2309cf096df5662f0a3bf7462e237f451b552601bf662cf56135cad5d7512",
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
