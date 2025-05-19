// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250519"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "10fd8307ae323f2b1f1e3f92c3dec2485395efe037336c54b3bf3475d5ad36ef",
    "sha256" + debug: "2d91067d3e7794af55737057bf2079d425711db92223ef23f74fc9f079115427",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "74d6f829beaf7643581a7ac4d8c49de8a577c6ca8b444f6ac5238d3fbd9a5c8f",
    "sha256" + debug: "89f34bb330e2eb5276276870d9c0b4519f0515ce09fa45023d34bd61655616bf",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1e0bb6eed942fbff50bc271a3caec783e683dcabc3fa40e2aea86aa5f4a41842",
    "sha256" + debug: "db79f7c2f50afe801fef34e9d6404a4e6658d447a9392fcf4352fe2535ccbe69",
  ],
  "executorch": [
    "sha256": "87883cdf7661a6d1e4cbe555c9a0fcc9adae1c0e81226623d46381196068fb62",
    "sha256" + debug: "25aba2924283ce9eac82d78524c30bc4853ceef26935468a235fcb888cb90296",
  ],
  "kernels_custom": [
    "sha256": "7732a32d800e13e8efee5b45c780f5984d42f586678d0bec72370e5310131081",
    "sha256" + debug: "156a8a70f83f09b59cbc59fabd3393c1aed7f233d577d1c6331313a30aac0383",
  ],
  "kernels_optimized": [
    "sha256": "4b43565d0d1d9e89a09dd0a7aefdf4e4881742330cb6faedb613925bf19935df",
    "sha256" + debug: "be3f806585c27d860d46a34550d0d73b36c8d51723e757919427c9ae26e087f9",
  ],
  "kernels_portable": [
    "sha256": "65851bc30706dcf9f9e139381d219a6b257c5798bb231dd43b4650b16e42fb9e",
    "sha256" + debug: "a9eb467fa33572eb5f024445f6a033d9fa59540b755a220f935953f95d4d7dd2",
  ],
  "kernels_quantized": [
    "sha256": "67b52eddae126560e6ad37b1088dc0ed05e9c0771b2264749b7439c24211834c",
    "sha256" + debug: "091c4bb5e941cf7a275985ecd380f7f75dd4788770886a69c6fbf83c87b3a1df",
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
