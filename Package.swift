// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241229"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "4ee29c5fc96c882a96a59280da3865a650e549a7d99ca9c82d4cd05ba0f65b4d",
    "sha256" + debug: "b66eccd3e074946b0c0b8a6fe83b2f1320957dc4b4486e82c0a367a8185558d7",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a5c432fd4e4e781826d0784315513032b41c9f84bea6adec3653ed20f46497a6",
    "sha256" + debug: "e922b94293de945c64d55b0f42362550e2d5d0679993e6556ed89f2bedcf3ec9",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "98bbd7eee000debd75a62d1500079858dee6105eeb1c756ed438f0f2c74ef7d0",
    "sha256" + debug: "204dcddb10e9e90e09133edc6ae87b73cdb35497300da22a13e58f1f6c55804d",
  ],
  "executorch": [
    "sha256": "b73d468fca999cda464dbd02d4dd47622633558cacda4edc7e19042ba6edbf32",
    "sha256" + debug: "e34bf984a79a13fc438fd7e5b51ff4ad66adc20fe990122cfeb8c1ff140219d7",
  ],
  "kernels_custom": [
    "sha256": "b172cf0bfb768d48015cd6e4af5b72366eeb64728dd6f3452d58e013aad88083",
    "sha256" + debug: "f46aa7297754b777383ce053d08fecccdea57674b58df89b59b4bd7168d15c7a",
  ],
  "kernels_optimized": [
    "sha256": "2d26cac0516a969440ed8d8fadb0f9bb79db303684c06c185c46bd8abe35f394",
    "sha256" + debug: "601c1731fc31069be02d7ea741436c6eb84b46e5fc298898f136d906ece3c662",
  ],
  "kernels_portable": [
    "sha256": "6d5856c4b47ed6718741b3d2679f58d70b58c1ef3c5013209fcb993be0f3dc65",
    "sha256" + debug: "d6bcda3aa741aa650bafda14fcc82ffa58d6642224cdd5d57a675139682bc0ed",
  ],
  "kernels_quantized": [
    "sha256": "8ba62848756ed0b0ff996573d86d3b8eaeb3ec06dacd32fb6907ab0b6f2be7ae",
    "sha256" + debug: "3ee1677f309c2d01bc3a13c43e37d2e5399946ccc34ea53eb3b379dae105e64c",
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
        path: ".swift/\(key)",
        linkerSettings:
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
