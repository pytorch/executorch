// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260725"
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
    "sha256": "2d2edc7799c98fa63e70fa997b27a9ec580f814fc15c32bfe13ef04c8932db29",
    "sha256" + debug_suffix: "835eebd12bca2d3212c8a2acc77fba97c6a6b2a61287b92f64ea3303aac23890",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "be9c2f3ce3d6a00f70462f35aef5b538b4ce7415a126fe60fa6fe942735adb35",
    "sha256" + debug_suffix: "7c4c93cec91c3e9f154840160853f083ca64d88982f636383db70a7022e02c12",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a3d94162f732dcb289257c3ba9005ad70a4e8c3c790c2ff927438436b69f7ba6",
    "sha256" + debug_suffix: "29feba752be10cc913022266cc3f05eb00d95b692bbbff40bf43722cfa895b1b",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "04a28577708b4d6ebac869fbe76824fb169a48c08556554b806917d17ec4f2ff",
    "sha256" + debug_suffix: "6d71f9cdcd3be903ed9b36f6ad854bbb0db234812058790d5f4d794f39ba632a",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "d6f1099244cf58e07b89e50ec30d12763cbbc4029312c573cd0fd808aa76d11f",
    "sha256" + debug_suffix: "6a893089f98d0c8eac16985e751477f444f0091fee5eecb783a0047d9f544526",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "da40b2bd983215fc66cdea889873905bc08b9a317d3d0c37a7e1dc3fe75c30d6",
    "sha256" + debug_suffix: "a90db5d4d2780776627daa89e51402855cb985f5dbe44d5e4c6315fa6933bed9",
  ],
  "kernels_optimized": [
    "sha256": "2e7d8fe9eb7ff6d126df452b896d061b06b93e448a9d2564424567954e5aa54b",
    "sha256" + debug_suffix: "cb2d9d4cefa13b1f65614deaf196d948857b76f94b1f33db87418cf9c1bec492",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "7d93c1ae6fb9827b18eb3b09798bb148e60002b12043e02d0fd6ee9a139781a1",
    "sha256" + debug_suffix: "13a7665e350462b160f39fe1b91f73de19c57cfdb15bd70ebe6cbc833ca4598e",
  ],
  "kernels_torchao": [
    "sha256": "2349aa99fd43bd436afa36a0a88eee539353fac92a9ef37ba536460c761b7cdc",
    "sha256" + debug_suffix: "fdacb5a318b3715e7108cb2af5224e85e532e8b9cf29b9bfc52f2ea846323418",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "4e93607802b640b7ea9a53910e657c45b9d252d55d6219ad9617eb428b55c405",
    "sha256" + debug_suffix: "74dca8a4193428d05a1b2729c16e8b90b46069ddfcf8581fc9fee45c120dd63d",
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
