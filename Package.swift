// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251003"
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
    "sha256": "925300542d3d59593244c640cb99fe8badb155ce114ba0e1c0c4447755f32932",
    "sha256" + debug_suffix: "c1965ed5ac1fba69b21267ca1d7814f8911aea64c9446e9ce84e677ad0ee8a1b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "4f31eb7b30e33b2e88d2b7435f50add7156ec8649fc834153e91c35b9247ad89",
    "sha256" + debug_suffix: "b41eae3b3f4d8dc6710f18f6582cba166eb8444644b961a378f610182364f9ac",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "9514838a043593aba6e995defb6dd9c2f98848b1d76aee39d62f6a36182eb224",
    "sha256" + debug_suffix: "7bc2be9fbdea0026e23d75c82be92b4fce3c5899966f27b93fa28bcfa6912d17",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "6c96cdcceacbb2820520d1aa31592f8987750cf13d1101941d42a2e48510e2a6",
    "sha256" + debug_suffix: "94b8cd011729eeb4b76fe5a0d48b0ee3bf9342ded8f8d0c935b4bd3968e4e593",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "3e1e846417f0b673a8ced3671e220ccda1d9c246320b14588f386d6e5c85cb6f",
    "sha256" + debug_suffix: "71032b117f22522757ac4fed5e705aeaf517c864a26e08887344c143e1d0a433",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "61869eeab715208bbe1a3f7d88ac955c2166fb6e10c53188f42d9f9b6894d2d5",
    "sha256" + debug_suffix: "475315159e150ccbaa23847d02f8a11306b859c030d4bc59a1a4d772e55ef817",
  ],
  "kernels_optimized": [
    "sha256": "b3ccde8ec1e46c43cfac90b658332eeb0449ff30449252967027673c0b41bef7",
    "sha256" + debug_suffix: "9ba3363740abfadb4d79b8f4a778f6893a52367b92233ef813211ee65c56d202",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "3e6f4245d093588c8023b1db7af957c28c267348e56ac1578d43136e3f3d66cb",
    "sha256" + debug_suffix: "367fde5fb10c9d97d4b3b311b180c52bcf22ef669633328bf9833879170b43f2",
  ],
  "kernels_torchao": [
    "sha256": "13f0365d63aff8d9bcb090c44d0939cf47640ea9d311b9435f889110dd1d4341",
    "sha256" + debug_suffix: "2955f542cdf854d98e402b060829e09bab7ec149592be6c9e18342601140da67",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "cc9c4febd80387575bfd00f760bb7de6273d80f491090d0d69df7595292dace6",
    "sha256" + debug_suffix: "4e2a93b2f02ee52c824edd19bd6b94062f833e532523dcfee0a1f6ba8de50397",
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
