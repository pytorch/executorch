// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250111"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "8fb47a8e3b83c5a9d7673740314e42aff162cd84e2179d27b5b90640bd2d6482",
    "sha256" + debug: "43e42891ad8770281ceb4f598fc528917b194b92ae39417519ca21ef8c9c8c19",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "6afbbff4e47601fec46b1e5744d6a88c24f1799d43cf292019ce20512a6693c0",
    "sha256" + debug: "1f3ec39e535e376f2f36b74fdcd530ddc5e615cb63979a572f74bc43f9a204ea",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "befff5aaa2009d7cb5f493cf35096b21b162d3a7344991d0391c58bfba6f7cf0",
    "sha256" + debug: "c6d2151db13638d70af89bb5c7c0d79ec9cce8634d42a67795d7408f8e6a0b86",
  ],
  "executorch": [
    "sha256": "aeedb16ea15ee5ab2027f94f9f7c96161953d1d9de687d364f37a2d3d11592f7",
    "sha256" + debug: "d84c2931a715071fca429480bd0192abe53f07128c2f7cf5761cef1d47b8b2e3",
  ],
  "kernels_custom": [
    "sha256": "b63d3aea1390a01a3a8176df55f37e2d5a35bee3cca4a673b2d1131de5e73637",
    "sha256" + debug: "c76d12f7a7d62998a89b2057a78837565daf18eb83ada316604f3ac8cc474635",
  ],
  "kernels_optimized": [
    "sha256": "27a5439a17b9df3056ed67d0a2d640d640d69b7ab2358d44524d05b10f81f83a",
    "sha256" + debug: "de3be50ae5fbfe81e642ec8a566657c23cc89f6419499fb4ef767379f535a99a",
  ],
  "kernels_portable": [
    "sha256": "5fb7b367946f1bb7207b49401597ec03623b2f17b7bf56829b76705798316864",
    "sha256" + debug: "2b09e80253fb1a1ff06f7b2ee0fe64850fdc4b7daf9573e64d531a4506be861d",
  ],
  "kernels_quantized": [
    "sha256": "3f8846836d562033fa67e2f75f52c7a5c94418db9da3aa29a82d75be3405900d",
    "sha256" + debug: "7db878e56ac1ed87286ca4cbaad77a61cdb08fa0ffbda304e054ca8a362b76d9",
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
