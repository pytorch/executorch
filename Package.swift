// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250402"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "77b37d2e2226f5378b57368bcd49fea365c4037962469792727aaefcd8037170",
    "sha256" + debug: "088f053d8fe019f2bd67309a6a19c3daa778f0d8f1cd6336f65ad29d41b7f190",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "6e97f1521a9551e565fb6ce7c32e2f9d5a67508329b233a7642521d4513e9270",
    "sha256" + debug: "901f673e912b318007d4e11e13c6fdcaec5b260a37c03b78256008c2048d8082",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e425f52daf46379ce4ca66982ec74f7702c5fe4caea3088523548bc5d0e37a8b",
    "sha256" + debug: "229bf7643ab9cd8ab8c3be716c78c3d5a5174426c03286d72d7fd43a47ccb984",
  ],
  "executorch": [
    "sha256": "696847bd3dba48a5462bab53243983e088cdbce84de723d7a4d6f51f76a108e7",
    "sha256" + debug: "8d633a1427a3dbaf2afe10ac2e09eb94c1b24e8528ba2240391a15859a8ef34d",
  ],
  "kernels_custom": [
    "sha256": "2d9cad963a80052291ac59944130752b4ddf55885037f772dd0525d60357d4cf",
    "sha256" + debug: "4806d6bc13ad9a4735abc9f1579e2ecc90239f34f82f7fa1e42f055149c30c2a",
  ],
  "kernels_optimized": [
    "sha256": "7d55a0c5741d5aa35345b1f9dcda1b61008a04fa9d4a126e1a7feea5522021c7",
    "sha256" + debug: "322a12b9b2e24d0e04b2179ef50c06cec1f462f2e54ee6fc11497caa787a4b68",
  ],
  "kernels_portable": [
    "sha256": "cd8b71c6ea9fb0107717bf1b57170784867bcf45a49b68a01003fed09d5214ab",
    "sha256" + debug: "8a3dc6f308bed3596182b81f7be3708504fae1d5b5dab4140e70f9dbe7a69645",
  ],
  "kernels_quantized": [
    "sha256": "b8a573ce01835dbbdd85268b5163ad44a58c38537c070f4e9693a718f86ed4c6",
    "sha256" + debug: "173caa9b48b3ce010d014b5559263f5912cbf911e07214f3049c773e94ee8d87",
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
        linkerSettings:
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
