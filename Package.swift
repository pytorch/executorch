// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241203"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "71c4b3378e5a8e48a1a91febc138e9f323ab2cb7737f08cd2b4b2a3d99259782",
    "sha256" + debug: "e0daf85caead63082c7c1502ce805cee29f9ceb8c184843cabfe1296b96565dc",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "40928561ce68c3d3b346330665c25f275e80ed4edd4c08802c233ead99c34b41",
    "sha256" + debug: "df29cc007e9d768e64c3e540086efe2264f1dc3eebaf4e0472b52044d5c2df6d",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ade96961fbb7b738db655deb1d89e8a73ce73f5f96904ae2ff7d0b153b4ec6e2",
    "sha256" + debug: "59f77f976c64265f133da66a3c2def4705f329be0896c5b80e3084b5f0a0b7dd",
  ],
  "executorch": [
    "sha256": "63b2f5d71a4e9ab57caf44f5d1124f092bb1c1ccb9c654b1129f87c06696ac8a",
    "sha256" + debug: "45cb094ab296555180cf3cf28ecfe50d4fda31a21be818ca3c365c69695e5719",
  ],
  "kernels_custom": [
    "sha256": "c4dfe5753c4e011b5f4c1ec22a37298427d6e2307de9a71a4a2a48869cb4bf7b",
    "sha256" + debug: "100e51709cd1804d9928ae1923506df5e1d561c8b00301b678424b3841673307",
  ],
  "kernels_optimized": [
    "sha256": "d7e36c9eeefac97c515a681a98b181865533a9dd1557e895381d2233399d1799",
    "sha256" + debug: "0869c21064ac5aaef91b5a3bae901717d68736c06e7dc411f1add9e12f7ae9b9",
  ],
  "kernels_portable": [
    "sha256": "2712d23e7a050dd4c9759efa2f570997dfed82b40b3b6636a25ccf084889c24b",
    "sha256" + debug: "5f2830598959369fcbb6beb0533b8bcef4f5e64ef989c165c55d409a132e62a8",
  ],
  "kernels_quantized": [
    "sha256": "d0b587aeba781f9e89fa7715ab4432ad08c5b88d5b402de6d3e94970b9f44289",
    "sha256" + debug: "274a029683cde472af75869e1837de38d4868ff0da745b114fba48cfcaf34c55",
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
