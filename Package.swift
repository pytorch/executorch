// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250613"
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
    "sha256": "fa2676e7c75b9555dd224c1c9d5f8082a2f983b34cec144e2b66959c0df1b85e",
    "sha256" + debug_suffix: "052599931d5d046acbfa5fce809a91a3faa35161a9efaa153b0cb17705ef0464",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "e4814187c4710893786e3641dfeb7a3a8ea4e4dc04ad9319c544972fd11fe547",
    "sha256" + debug_suffix: "1d90c53a6ae710292c29729da8011eb97ed214a35e0c8c9c64f8aad2f2e0ea1a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ba49e0e34f4d7fb0d5acfdd7ee2531744c628972046cb6acc2709cd86d9a5cfc",
    "sha256" + debug_suffix: "1b3344d5f14a9fda9f86e7b670030dbbf947d9321f9d410bf687fe9580c71df2",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "fbcf87027d76009ad8e0c3329179012742b087c3ee2e82b5ec06350cddc92d27",
    "sha256" + debug_suffix: "e7fd9bed33d78afa5388b06a7aa16085e3ec4aab3e0dae9d21b0c6c4e7277887",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "fce30c51b72c38aaf82e286db542f3ce12b0765cc04f96ae9c00a8157e9e0b6c",
    "sha256" + debug_suffix: "b8df2b2cacde78f04ba240d6358ceaf940a48cd0193e53467cf14ce248a1a517",
  ],
  "kernels_optimized": [
    "sha256": "552813277ab3463f3497646164f41d9b588821c301d2555a188613b587a54f49",
    "sha256" + debug_suffix: "3713fe3dc2e1141d62ab4802c3545e57a50baf4e473067f3d7d3531178c36df6",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "e3a0cf404b4323d5d1dd5b1679e8a108e9a1732df6dc78ec159de392329886be",
    "sha256" + debug_suffix: "98dd87b2e396bf1b703f6965d50dad5e1e9d884c24092a5e21c4065c2b883550",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "1e782f0bea878ef09e1ae6652cac236d8f595684bbc0ea9ca1f7e77a2a5e4352",
    "sha256" + debug_suffix: "f02819ecabd48880cc8aa25d7e1890336e36858fc17d826df0c378f0b2b5a901",
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
