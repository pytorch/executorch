// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241119"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "fd91834cf7a14895a5daa174809c5f5f53adffae37854d4e4f32bf4c07494021",
    "sha256" + debug: "741d4f38f65b8e4db7dd46702621c30ad1d5aeb70d20678b637d69c03b7f429b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "295f2f6c7e702ad924997baa97dfa5f449e32b66e5ed275205ad634b4b9a7d86",
    "sha256" + debug: "0909b3b5da82ac00d46231db8308836c074fbd075374f47627e9b4a4f6053afd",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "794ba328060c7ea7bcc121541e5fca44ce800939ea3b5020e1e22258824c6043",
    "sha256" + debug: "8ee5e319409f731abfc6f30e36339411d36d43909c2d87bf3798c3b8c7a4a729",
  ],
  "executorch": [
    "sha256": "a21d875948cecdfdd128c3bd69d4e5c3eb15ee412a857f106f2f6699ebd4adf7",
    "sha256" + debug: "0602296c5c29b4bb1e5de6cc7ab4f7de412f6446ab66679a952d62e5ac85caa2",
  ],
  "kernels_custom": [
    "sha256": "2c13c9809832bce9f5c84c61616859e7758187f95bcd7edf6a23e7e70034ddf4",
    "sha256" + debug: "59d86fa16811587a7397394a3f737398e716261133cc60b8b715fdfc2ef4e426",
  ],
  "kernels_optimized": [
    "sha256": "74c010c52cd7b385b9a4951a96c77343f48c403a4a28e1f03dcc80dded3bde39",
    "sha256" + debug: "b8f30c6efb013957fc0054bf03c0317634fd1bfd17316b41abd43d447a610f5f",
  ],
  "kernels_portable": [
    "sha256": "f4e53f9b8a6c0a37aebe2a68fab2c10fcb43d019f273b28519ab06457ff6860c",
    "sha256" + debug: "a7c210f6dc3c3cee043aa849a1936ca1a8e07086ffdd2279d7fafd109d8248ff",
  ],
  "kernels_quantized": [
    "sha256": "73874f6e203d959f5caf873dd61d046028a25399e41f6b41d60b53d9317b36c7",
    "sha256" + debug: "41b0e2ea215b3e05f9f5a47ffce403124226ed26e1f527bb4a372b50fa438e38",
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
