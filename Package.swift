// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250502"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "0283dcb54c6aa4416d91ca3120fcfd482ff6e95d96b6aa57aa0d6b2386c18e6b",
    "sha256" + debug: "d68e6abead7cbef9036e189e6f7215d8190cc41e5dd3f7304e5aded543ac9d8c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "112f3a5859c30ec21e1c083ebbcd0ce30c1f8dafa603efecf7d823dadd5b4bd7",
    "sha256" + debug: "7dfdce03a0887fa10c8750b8355ae2754c76e74f47a35c6633bd83344e2490d4",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c66028de3e965faaf30ac51a4f72cf5ff99252d463431793e3a24f399fcc46ba",
    "sha256" + debug: "84d0091302e5800019ef62bfadef58e16dd02fb9322f2756166132d0d0ed4b12",
  ],
  "executorch": [
    "sha256": "20fcc51afbdaa1a39e3224a348c03418f6d5f37fb13a248a718215e4e436dcc8",
    "sha256" + debug: "bcadb73a73a586ce2a4dd5ba79ada9d2d226ebc8f40c8cb478f9c3c71e09dcd0",
  ],
  "kernels_custom": [
    "sha256": "429ee811c0336208fdd73bc9ecf04d318d39fde95f578e9ee5a7a4ab13bf0106",
    "sha256" + debug: "3f1b78b1d48119b9cc7dd7e648f36412280344dde74c5d2eafd613ee5716c1a9",
  ],
  "kernels_optimized": [
    "sha256": "278999c8c875338a139cea2502199ee773c74d1db0bc3e1d39c5a3240fbb3a4a",
    "sha256" + debug: "d34ff2495806718639203ec46db091f7215bdaf7b60c43eb95e14f1ea5b6f865",
  ],
  "kernels_portable": [
    "sha256": "fa27008eef87853d7b3fecee90ba573a72ac0a6624cc19cfd03a0f723948e9a3",
    "sha256" + debug: "5533087313acb8c9726bfd26287569938931533635364891db3f3be9c2e60e50",
  ],
  "kernels_quantized": [
    "sha256": "1b9b58414b1061896f2d5afcafd8486658b6ea3309913c901a7012cb9e802b4c",
    "sha256" + debug: "a65298c992856c2809a8a49e12729009adb1b33e59ea8f47da46ff51b49ad9f7",
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
