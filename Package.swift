// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "9204ac83a8d29bd1c1e3892dbd3cf726165612ac810b46e5cfa5b29fa4d54cd9",
    "sha256" + debug: "d5a25e06d3d12a543208bb3b5ac2399ddbc9a7d4de3331fb2deb46c77381b1e5",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "f6bc606768bbf4f4977c7e06e2d96cdfabf5100166af5115230391ca47028814",
    "sha256" + debug: "a787c6ae54792250d85b194704f448ecdb860c58c9b9b9d141181391ff6cca60",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b7085fc113795ee8f58ac91d02cde071bb7ffe9f0ebfaed5e09b6ec99d2942b7",
    "sha256" + debug: "4389b44d377682e9d66447fc672c8dce37937a9d4df23f3a86abaaf40b9f101e",
  ],
  "executorch": [
    "sha256": "ad691741c75243c43bb813b8a3901ce368e590f20b2329b3e96e4fe49c2dec39",
    "sha256" + debug: "535a53704e34dc472272e9952e6e4c5dd0ddeaefce9251413d650053e9a2f83d",
  ],
  "kernels_custom": [
    "sha256": "770a56c364eb9347b041f3a13dfa73ac69ec84bbc79782895c95c655d1d1a609",
    "sha256" + debug: "9d40846b3eb120b9db26579addbcff264bd20683f267f6f82602556ab661d7eb",
  ],
  "kernels_optimized": [
    "sha256": "ee9149f4ee9f8132591a06aabd6bb1295a1dc548dacabfb1d97f9cf08c5f6932",
    "sha256" + debug: "6540895cfe95f8911cd1380641a825158d7240a5079c3bd683f93cd18da572da",
  ],
  "kernels_portable": [
    "sha256": "6b80172a7f97e973531ce3321df5855293a0fcbc909e346f4c1e77476ed451ad",
    "sha256" + debug: "9a10375d755735753c79bf3fb52d7d49b56c82053a2a5290688bca3c0e16a3bb",
  ],
  "kernels_quantized": [
    "sha256": "67e0e27af6227f3e20a0ba37ed32bfb8f45c2c2e99129ca19672da38eb3128ed",
    "sha256" + debug: "f04c7626536675d9c68a4ed2a15b6f499103fb0ae0764f0be33c0bb35baff15b",
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
    .iOS(.v15),
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
