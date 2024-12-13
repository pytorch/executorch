// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241213"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "4f504254c93e73897be061bc7e5b1389f8b570d3116edf70fd35f28551a55925",
    "sha256" + debug: "00910d4aa5a18a1e74411538a3cbb0173a33ce2c9781bfaebe4d3a0aa501d8b2",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7105a884a2c5b4830d27ebabc2a7cdcd44b79f131363fa6c676337dd8b59317d",
    "sha256" + debug: "c0678ff9b150f5ef17d6e0200a11914cb65980b27327900a8726b457d9904f79",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "82d0fac744661dfe885353d30d0542bb47a0a0b4da0b4e7884029744544b4fe7",
    "sha256" + debug: "53d2b76b75392929c674dcca4de0e9d7d889a6b63a27b0a3b4988ef0e545ac3f",
  ],
  "executorch": [
    "sha256": "6dd6149bfcf90d553ed558f1f09dd2f7534480bc833a5dc382659e8676ec4869",
    "sha256" + debug: "1d3380e6beee2616daf26484905c09b7b258b7fe0329103888a56f319e4130f4",
  ],
  "kernels_custom": [
    "sha256": "cdad7a26812f776f6bb752d51593425e2ee87feccc10c6f9abfcb003587f3ab6",
    "sha256" + debug: "516951ae255e186c0896d261cc199435cac4f15878e3ba6ab86c454e1106c2f3",
  ],
  "kernels_optimized": [
    "sha256": "13d42a7c71ce5515e5bc3596e55f51f82d6eb4508ca827303b5b463c2907e07a",
    "sha256" + debug: "fa43f19f04cc869fc16d51c1af0106f47b37ad8b7c2b0d4825f14bd1c2acc618",
  ],
  "kernels_portable": [
    "sha256": "6df0e53d809c43fa624189ba541f055fab26ec3507c6f56fa3c3c5771ce1e1fb",
    "sha256" + debug: "1066138cc0f37d28b510b3ad3c82239fe0fb42cbc2233abb249b080d1b440538",
  ],
  "kernels_quantized": [
    "sha256": "f299a1f7ab5fb0d280b3a93d9fe58b8e7182988106c0ea57aad019659572bd7f",
    "sha256" + debug: "19d1619ce9cbcf5d001815c15ee6b603bafcd7765bcad6aee93bb136d5248646",
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
