// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250826"
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
    "sha256": "e840387103a7639f5ee4f2f7af9dbc09767fc8c029be29a214b9fd890d1a1522",
    "sha256" + debug_suffix: "1a858b778d27a15059cb6b27c74ea8fb6cfb79800420f71da0c8d467ed555eb0",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8bd60403f5b0ec31fc66f0fb428c6d81328353f5a3e57518517ec94a8ca3f21b",
    "sha256" + debug_suffix: "1f90055b942eb1a6bee40bfe8f96e0ccadc40f987b755fc784b439086c264622",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d9b73ea24daefb54e0e27f9900e61b8cc16621d1bd8965d58bf35a1fbdf331c8",
    "sha256" + debug_suffix: "19ff4f028a58f1608e0c4fa56dbc7311a6b4eff17a7f87b5c8bf885c6600a24c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "233869ecbfee072dfd60250e17a99cba4d538d0ae5c5c9d417e4636a5c86d1a7",
    "sha256" + debug_suffix: "e7ea34dd166d0d2373f8f7afb879ffd435cd67cde58fd52d90d9b04beba3ea2f",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "1c9351480d8a9fb1a5d10b1e07dffd65f8de475e14fad65868edd76c94cfc10b",
    "sha256" + debug_suffix: "bdd18c6088c3875d8f0da1cfbdfcbd4aca59e8f826ab551f708ebea2b7ca9d6b",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "93773db5126a77db226b4db9f790b5b8e7ce930f4dd617f525dcbbb13ef8875c",
    "sha256" + debug_suffix: "cd0ce210b4dbdd767c562af2873061780f60a983a56fbcd795a11eb2e2efe139",
  ],
  "kernels_optimized": [
    "sha256": "86f3ac4a42db4bb8154945117e5074037f4f3b6f1263ce2200eee6ea31d4563f",
    "sha256" + debug_suffix: "be021580e69237a79b55ac1227f231374857cddf37b3f1492fe3dc144e0c53e7",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "aa607823f20504c8e92055e20f17e6b145971c0bac0732ccc3ad326b8fe3a474",
    "sha256" + debug_suffix: "b8ae7920dd4e32186d63e8d0ba075adc0bfdd08f96927c9295c616520d3dd315",
  ],
  "kernels_torchao": [
    "sha256": "__SHA256_kernels_torchao__",
    "sha256" + debug_suffix: "__SHA256_kernels_torchao_debug__",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "bbbab12d4ec35eafb22dc5c66525b5eca4e8faa3595c2f5f32ab1b39ff16b81c",
    "sha256" + debug_suffix: "85ea5976347b30ad196adab5f18eea05ffb692383e4d7a95ad06ba2d8a347dee",
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
