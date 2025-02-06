// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250206"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "76dedf825e772230e3cd6835b1df783ac452932221ac7710b7ab25db2e098669",
    "sha256" + debug: "226a9acbfb585b044f96ee602aedd5447ef31d33f2c35fb04534636e32115f26",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a46ff718eba065ec29eb51b60d5120e7c75a31a75027fc2b58d761d24456f98a",
    "sha256" + debug: "7bd705298c35cf31450808aa5aea0b33d48aeb3dde743d401e6ab83ea751b207",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6ff1ffb3ca8384839b2d6566a44b77c52f771f40f488f80bccf3261e7d9701d4",
    "sha256" + debug: "1d6637ad36a39d902c7e6327c2fffe9b2c06e59e63fca662ca153e2e67e193d0",
  ],
  "executorch": [
    "sha256": "ef0fcc7a6f930a04454d88d89dde342ce1b186d07558c332a3f85da5856b6a72",
    "sha256" + debug: "7a7f29bfaa293aace45843306bff4f828d0430828bef72fbaf91f2b8b3b6aac0",
  ],
  "kernels_custom": [
    "sha256": "c341c2bdb05848f1b593e94490b885be369d048abbab6e8081ef12a3166c9311",
    "sha256" + debug: "86bd2bb09fec20ecb66f7ef4d0eb19499ea26435cabed4eb730cbd2e5204442e",
  ],
  "kernels_optimized": [
    "sha256": "53a56181adc844336d7701cda3b964032ec071dfdc3ef63a4174a5a81842c15f",
    "sha256" + debug: "4207743827ba8953131a771e6c8b3061fe8cf2f2c93bff127fdbf9f830aebde5",
  ],
  "kernels_portable": [
    "sha256": "bfa7ef99a2018be2a4588fc00a88b1891881cb0eab42059f75501fc2d684135e",
    "sha256" + debug: "ea740dba128b0ecf9542f968d633aad3d622f5aaf777dd5845138b1c3a183f7e",
  ],
  "kernels_quantized": [
    "sha256": "64b5794c227eafa74f29970584cce7aa04f6fffb981db3530eef21ddfa9e8791",
    "sha256" + debug: "25d9e7eb1c3fe5fa500303213c04aecc1f9d7ebf26448d8e619d034d460d21fe",
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
