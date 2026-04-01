// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260401"
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
    "sha256": "8423142a4a72fa57e63e628d870fddcab08b7305ed21fe3d357c88a7aeac27db",
    "sha256" + debug_suffix: "75287b5ed5e7e8d3a050e32236bb466ab79386ea9173af8e0477f2ec4d4143b6",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a0753d9e4f41115f08d366b14d16dbf1df0c4cd0141729f742259663fd472f3b",
    "sha256" + debug_suffix: "603dcc370e55f58e8d0eade3b25b7062c21b55c27477855b6523ba41d7a4fd33",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e4df805d57886a5def7630327b8bc5de56e42e51b16b7972f3afb371c8ae4ef1",
    "sha256" + debug_suffix: "6936650e060578ed9d0f2302db12e65d19d94f9d44389be2c13e6ab46673231c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "38e7a0fe6e83031af1657e86d75dab20be56098b68aaad1d1312cd6598b3ad5d",
    "sha256" + debug_suffix: "ef453992c3bb7f000278a2930f2cadecfd72c259f3cad4d5fc698133a2959a08",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "64b5c75f632836d889dbf68fd59a51de1aa18d6de3dd683e2811b38c702ba2ff",
    "sha256" + debug_suffix: "14b6e7eab825264ea3c8326f41dd3d6fb8ede30947e20d27b6b44e9a545415c4",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "a5de634788a0dc6dec27668c595b63af975e7a1cb1e0cdcb8cce501795e09d88",
    "sha256" + debug_suffix: "d52f13c0fc260cbf471c67ce814a49654d15acee81d16c6326ab0df75df976d4",
  ],
  "kernels_optimized": [
    "sha256": "56dddd89d6f8a6d10682d7b03c31104559711c6c6573fca718c2784649914e64",
    "sha256" + debug_suffix: "657b101427b235e98b4ef589f6983f47799e1d16eefd14c8dc9de2c2a3b81f72",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "4df51a6fa291be3df19458fb931aa4b665ed2f193e07ce3030a9a228f0d67a1b",
    "sha256" + debug_suffix: "0927c550c458c370b8c0402e0cd224d1966aba36961a24f44453e796a17088ec",
  ],
  "kernels_torchao": [
    "sha256": "c0011ac89b437c3b854ff346245f3e0bfd771fb4f761ff13961855cf5832224e",
    "sha256" + debug_suffix: "9d792b5eee00d3609db800d75c095a3e29a43c22f808d42cf6f2612949fd1f32",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "25b50040a85412887b274ad86efa6b81f9260ddca29eed45a499855aa0a2242b",
    "sha256" + debug_suffix: "0eccf1e105030a9333e176943c54dc3c3e973859d35bda3b1b242bc1c4fe53e2",
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
