// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250330"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "c48488fddc3b60d66e2edcc070ea6587e5dd4bd6dda893f703e9dbe75e145951",
    "sha256" + debug: "e741af2e986db7aff5883bd54cc1eefa532ba7b06349410dad54723aa3ded2bc",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a9a225bc35fff1a6c8b551b6a58b05a28b6dfcf219bc998072c388b6be790245",
    "sha256" + debug: "646d4adc572e4aab6ae553c12fa09b6d698abf7205fc2a41302a3ffbb04b7e1d",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "53c86edc9f01f5f520362b3a8635e9e73b301b52348c2b633dda51d9c329dfc0",
    "sha256" + debug: "129f55135896b101b45084ae70d7456d86606e697afc528a35669c93523c784e",
  ],
  "executorch": [
    "sha256": "0311902403157786e2d856f05435f8df1475cf5e8b67ea4bb2e2dc43105161d3",
    "sha256" + debug: "d87c67b3ce5690b2762294bb80b6c52e62d0a4c5da05862a25705b06dff5325a",
  ],
  "kernels_custom": [
    "sha256": "8a67cd03cf60115e8d9a7e1e76aa8b99cdc95a53adc12bbbe82008e3a883ab95",
    "sha256" + debug: "f1cd9803859aecc777e662bd2579c0c4ccad35fd11f0ffadec41ce9907277383",
  ],
  "kernels_optimized": [
    "sha256": "7a7314fa94cd830c8987d7ce4ed6bb7d70b806c8f1a1a4bc48533e764aaa829f",
    "sha256" + debug: "8c39c84eb0e02a79228b064fbb2bebd21debbe903fe2db251d13844629f16e49",
  ],
  "kernels_portable": [
    "sha256": "08b90a7fc548ab6bd6da9ec7b65fcaa2de2c91410b64686aff967e064ba230c7",
    "sha256" + debug: "49bf6137e48b4b875e067ccbe590d60ecb8675b31f5c8153c5818b27ad2b17fd",
  ],
  "kernels_quantized": [
    "sha256": "02482639ba3576d5662f11f3611dab28792ac385191df9c3fbdcf06baf859933",
    "sha256" + debug: "f5193f32f9eec20015717a788219b3d0912531c5b90c35a8d4c02fb5beb7200c",
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
