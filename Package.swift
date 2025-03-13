// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250313"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "acc3509e1e5716fe4d355a249ae677095735aeb5f42f0efd14d5d5a5faa1af91",
    "sha256" + debug: "8e30a0ac1ceb0965fd5f521ef3237641e2b0617ef4e00db1bd54de3b0f0db869",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "78e22e9f92093b0ae277036122ed5a0a17dec96ed6d794411a6d73142c90f2af",
    "sha256" + debug: "d285d9a30587d30b9e7aed2c8794a9f4c4b5c755bef42c2aacf2665fad4f0b34",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "9a377bb340c40851695f0c6a5222fe31513343851f525b51e4c5637d45d2fcde",
    "sha256" + debug: "ff711367d04e9fe2564947ae6e64189be4b71a1900fdf7068fca8a3d909bb889",
  ],
  "executorch": [
    "sha256": "7d279f76fff6d69e8ceec9d21e1df6c5d77408dd3f20d9ee1640c63abcf03e38",
    "sha256" + debug: "fcb70bf10159ed576a7348330803302816428d270dc35768729802194bdbf3fc",
  ],
  "kernels_custom": [
    "sha256": "82f304c81ef000505997f509aba36240b18b0a1d6f31907436cb6715da5c4570",
    "sha256" + debug: "70588bd06885c212e3117ea8a075267a76e83b292388fdc4dc4d9070ac4d7499",
  ],
  "kernels_optimized": [
    "sha256": "5fbb738097403f7cf5e2a1f6a603a4cf2cdbc319d36e062317754bc830bece80",
    "sha256" + debug: "dca8b56cae1d0d80e3309c000122bfa2ba4919bf1fb52090e79db9c2f5ac8fa0",
  ],
  "kernels_portable": [
    "sha256": "79636866a9b4c9c5e3c1808cb2ac60fc2a8fb625b8bb107880a3511683811768",
    "sha256" + debug: "0baed7fc4b7235629643353f55ddb1fac112c2354b6a89f58072816c22d3cbcc",
  ],
  "kernels_quantized": [
    "sha256": "69c0b694e094803f8ce68efe6bd0d96d527273cb0da71ab0e2506b88e0ed2f53",
    "sha256" + debug: "df5a1d46a4429157d6793ba8e46ee01bb71896d87acc203351b90b11a21ceebf",
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
