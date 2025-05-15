// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250515"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "094c555927454e36e411c07c393a91fd02603f661227b1fdbfa186c6de925933",
    "sha256" + debug: "8f5c0d1071b7b38ed2c702a8b790a66914f3352298d85eb13a8800328a2f4b1b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7b3204a8d38796f768f82f193ae6a86fcee8bbc9af327c761641bd40c746c3de",
    "sha256" + debug: "a5d11f3786f9f5ab4ee473134f5f3bcd0dc9bee0410d37949e85f23f5df58714",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a47f8d71128f7830b0dcb314bdaca86420628cdcddc38bedb2c94cde3dba361f",
    "sha256" + debug: "5990d6d327061172c44c677de1132ec1d09d84ac0c16ca055bc25e939201b25e",
  ],
  "executorch": [
    "sha256": "1fe50540d7cf918ed981b38d9e723f31305fbfa51967eb3f460f631363a401aa",
    "sha256" + debug: "c5e6f5ad840205b7b642667393db159ca503718238ddb1f364a417dbb001ac8d",
  ],
  "kernels_custom": [
    "sha256": "3f22d595797dfee43f893a22c4def4c8694847759f0ff8336bd534295129d925",
    "sha256" + debug: "d4d05c19cddf1677cb25d28c921149e200bacb853b34cd333c677b1ca2a48a9b",
  ],
  "kernels_optimized": [
    "sha256": "686c974a7e77591d136de1615a9d0d7aa49d51b5b6dae95e7fb12818ffe2cd95",
    "sha256" + debug: "5e1004ea908f3d9f2494ab43594d740fb9fdd9ad778e4bae3e05151ad68220dd",
  ],
  "kernels_portable": [
    "sha256": "ac6cdb99670e76797665fcad759a7241f3e59f16f7083bd323e466399f5040d8",
    "sha256" + debug: "ce4c3e8b3cf2a53c6543e3f475d2c19f5730b9b7ef5f6796d8705bbdd2f1e050",
  ],
  "kernels_quantized": [
    "sha256": "561ea019166857da1ca68b7ccd9d4c72960bb096896d912c2f1d0d02c51b9f69",
    "sha256" + debug: "9e8df53acff5b9b7b3c74b7fec36a03388bd58c2267c06a18b51c76a6a40c927",
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
