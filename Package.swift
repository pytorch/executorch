// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250520"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "7c0fb2d0d3fa96459e092f3f8092a6497877992aef63a1cb7193f78528be631f",
    "sha256" + debug: "6861beeadcf6178ec221ec27ac6ce8718a615a006e979af9c47ea9a031675697",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "595ee619fdfabf6494670c88e0da00164f7116136f39094832d2a80e95141c5e",
    "sha256" + debug: "a55d7c10d3d6ae0fe271842b7d8e5fb03e410d3bb3ab3a4ef175557ffafd7373",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f96052e204f42946c71e05f832a42d693abd8ab6b297f13af19992e251cf3185",
    "sha256" + debug: "e2ae1b66dcb2bc2f52dfb73a259fe0db53bc36aab06d40869f6c58b0b43ca716",
  ],
  "executorch": [
    "sha256": "8dca96509060f1e4949dd57e7d427336bf0231fb539ef67b9c112e93403bbea4",
    "sha256" + debug: "e87547eb5e56698bc2f7728eab7bef456ee5d49a3ce6fa7153abd256668203d5",
  ],
  "kernels_custom": [
    "sha256": "80f30da1b82cfbdf0de58f9cddb8e00a1596818b890e5edb19331fd2731a26c2",
    "sha256" + debug: "e296ea30e9686fa162daa16290291f4c907a36c628604630e4e09a654c17944e",
  ],
  "kernels_optimized": [
    "sha256": "ee9943380086781a2db15e7f610c655e81f07041bb07b6b2748310d9f9b76b83",
    "sha256" + debug: "884c3b6ba46c23969863ed9ba45ab7d96e45e3d3e8a2b61882a83ca8b2e233f1",
  ],
  "kernels_portable": [
    "sha256": "6526b717f9cda9da759611aa65213ac3c76a1c1416d5ce1dc253d12e211ecb6b",
    "sha256" + debug: "81c8a80b62800fb8c25be478335d8ef0f5226bdef0a62501eabe7f91161209eb",
  ],
  "kernels_quantized": [
    "sha256": "9fed7416ffdd71879bf5d6e7196bb334d5902627a5ce67da3468f71f116a17f8",
    "sha256" + debug: "493ff13b0eee93f16f7ad7c652a845d3978968db24191f306e7e6ced144e839e",
  ],
].reduce(into: [String: [String: Any]]()) {
  $0[$1.key] = $1.value
  $0[$1.key + debug] = $1.value
}
.reduce(into: [String: [String: Any]]()) {
  var newValue = $1.value
  if $1.key.hasSuffix(debug) {
    $1.value.forEach { key, value in
      if key.hasSuffix(debug) {
        newValue[String(key.dropLast(debug.count))] = value
      }
    }
  }
  $0[$1.key] = newValue.filter { key, _ in !key.hasSuffix(debug) }
}

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v10_15),
  ],
  products: deliverables.keys.map { key in
    .library(name: key, targets: ["\(key)_dependencies"])
  }.sorted { $0.name < $1.name },
  targets: deliverables.flatMap { key, value -> [Target] in
    [
      .binaryTarget(
        name: key,
        url: "\(url)\(key)-\(version).zip",
        checksum: value["sha256"] as? String ?? ""
      ),
      .target(
        name: "\(key)_dependencies",
        dependencies: [.target(name: key)],
        path: ".Package.swift/\(key)",
        linkerSettings: [
          .linkedLibrary("c++")
        ] +
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
