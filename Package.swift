// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250704"
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
    "sha256": "fcf3d4147abe1a237ada61bd3e2789c0bda870ce24c4c59de2ee2ee05ab3b907",
    "sha256" + debug_suffix: "4581b99c6fe0d22b637ec11d04420f1aa8834d8cceab691ce947d9e0e3a09ae2",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "46a3163cafc15e74a5ac10b9ca60c3370da83e84766e89531605952abe17d991",
    "sha256" + debug_suffix: "256483b6d81ce2e5ad7ced4965da6ae67f43ac0309af863021dee6b3c1553e59",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "202346cd542202fecba9092d707b250fe92ef888a817d5922ceba84544ebc5ca",
    "sha256" + debug_suffix: "5a8c4671da4c0e264e8213ce55ac9831ed1451ecc821c850610c7832e8ef36c3",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "be2bd7da04def172a4b4079b05ffb2d81972ce29f83dc8643e2a7c8d03fb686a",
    "sha256" + debug_suffix: "4adc0c19ad5e14eec4adafa25acf43dfa9ee8f8e26045bc2fd9b85ba4f51a75c",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "9ac6bbc47c4178e9df58ba61899b9ffa06fcbefc2dd4fa777e4d150361bcbd64",
    "sha256" + debug_suffix: "8b3ae43140037bba472c8c6c1573eeaa7d628411f22f21af2960a973d00b76a0",
  ],
  "kernels_optimized": [
    "sha256": "96164a19369aed3f3f44ec1bbda6fd6e907a0f2375ce83e897450e1a2c69372a",
    "sha256" + debug_suffix: "d8685df429bfc662c6e058805b19645ed5e8a93654331d359368095c9244d682",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "eca74cc039df9ba19691067b8ca6125a137189fc3fcb5633c8cc926ddc3a4c1f",
    "sha256" + debug_suffix: "7892f792f0236e89a7855fe7407e4425d1c6827a96774669f5fe5363abca1ed8",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "aa4acaec655a6c55e2f8e2e333af0d584400ec5cc7bfbbc20a02c0a851545aaa",
    "sha256" + debug_suffix: "cc47ac68c2a0ad4f6cd07c7195616ff42736f87889ede9f6b4037c67650e196e",
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
