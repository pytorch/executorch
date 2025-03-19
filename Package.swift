// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250319"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "f3f445c86b4ca5e3ee7b1751e770f1a278e58270b9042f4a83b42d7a100f5aea",
    "sha256" + debug: "4e6c8a5af86e0085536907ab1b13c879b81fd01c805f0eed77f04530e0701797",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8d419bbbd20782055ac79451ccb0d564a68084358389e8dad9f464dec98acc0a",
    "sha256" + debug: "12e129f7e83f4793bc193b53c9a61ceec6cf97a70ce92451141f4453651db4cc",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "4d935d2dadd5dc836746e605ea4f7f17c039f3d91b3d3cfacfd8161a2e777345",
    "sha256" + debug: "a24260fe5c766245b1e139dc3c3dfb9b25c3263ae1fb9daee30af84359d480a9",
  ],
  "executorch": [
    "sha256": "b126c3928554af1387c8837b9123d7d741066a716b38c98fc72f1e0e81d6869a",
    "sha256" + debug: "ad916abcc8b3131b4c5de6530fe6896dc76faf458d97ebddbeb19d09092e6911",
  ],
  "kernels_custom": [
    "sha256": "db6a5ca6745e8d5accb1145b57f2cdff2e50ca2a454fc6f27c97dce680c61818",
    "sha256" + debug: "195b47ace785a6c135ca0371794559123eb433ad0e3f2d21342a305e816bd325",
  ],
  "kernels_optimized": [
    "sha256": "72077e82609cb3309a79c6634ce8492d6d51b2b33bc0789aaad9a3c04fbd582b",
    "sha256" + debug: "f3b346ce89e987d68188944391d55b5418c57275034ba1ca083f03b076886796",
  ],
  "kernels_portable": [
    "sha256": "82fd1eb51322b099fe5046a3fde5c7189b953eb4b982c2d82905275bec1c0722",
    "sha256" + debug: "763521cca1b69badad493c59f290816ab5efc4fb0bb9f8a7e2a895ed83b14a74",
  ],
  "kernels_quantized": [
    "sha256": "49c9aa68d5e247ab083975d047624d276a72cd3d0aef5968d0f764a184e72ec3",
    "sha256" + debug: "f6003a1b2c96500097bdee5e8247a59d14eac9ad1054bc24c01d487770f7b427",
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
