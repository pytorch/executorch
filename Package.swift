// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250311"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "5521dc96f2b2178c8047921290a4aed4d353bf14ff59b5154112ef7f8fe03a89",
    "sha256" + debug: "54b56ef5f6422c89472d2efaa17e74fbeff60e8d6cfc904494995023491e7a2e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "c7fbf553c6045d84962d3c980e102b4b502bc69b3b02c2787f1b211d2b2a450d",
    "sha256" + debug: "9ad51027fe03f9855d040c8814b2a8a0f9ac55cf7c7aaadf215009d5e50045fa",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1946cba8d83a77df6b5108f4b53283c77dcb87de8af2cc256ee20a9408a54221",
    "sha256" + debug: "dbba01eee859b9d7d83104a9abd0e0a9e68f79b7c97f9c8c3c6766fb5716d0ff",
  ],
  "executorch": [
    "sha256": "079190952397c425c0a0fe6ce2b67aa3a3b07b89b6bfa50d19222cb2cd3f5c60",
    "sha256" + debug: "425cef2b2c31c73ae0dc0ef2b341529b56c4987609623aa223eaed6862d7cffd",
  ],
  "kernels_custom": [
    "sha256": "1384f1b5f04e2f3e4b20103f731be7a8340208e324d92458382a4211516f5716",
    "sha256" + debug: "2cd8d4734e458fe357d348aef736cb61a785777179a7c9c1a3763b6b38867d4c",
  ],
  "kernels_optimized": [
    "sha256": "f5737f299a72361bf08820dba57ad3d9d6948fd576c9a94bf1207b6a4ab164d9",
    "sha256" + debug: "1093a79547c18f2dfd06006b6bb8a48aef94f78b901b7dbce070e4a03b4aeb17",
  ],
  "kernels_portable": [
    "sha256": "849401039e66b81a15677f410667914dd1684f7f14bf73ed397a5c26ff0b6079",
    "sha256" + debug: "aed7d15b227d4498b58b829c78e5f3463caf16885fd007ca1242fc3ad96aeef6",
  ],
  "kernels_quantized": [
    "sha256": "43c4fe4a0771548bb316775b9923d3ebe52246ae3f4fef1f50210a1e7def667b",
    "sha256" + debug: "9e7df4f7cef5fcb847f179cba547035d83e142480becc103fefe655d3928675e",
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
