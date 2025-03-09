// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250309"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "f3beb6bc12eb43cd232f5dc0736b114cc42302e540c6b728c152c7d7d69f8eea",
    "sha256" + debug: "020d95520f4cf4ab6060e6af2a7ef082be063bc1e1adaa3ddaf946636783a21d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d496bad07348f603dbaa4a9d844ba557b0aa74adacb7b176fc7da401c721b04d",
    "sha256" + debug: "05af51c2cf1e974256cd44c4345eb69d7e2e83229eb7c0fdfd852db9e3fae0e6",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "21bcc1d36275372121743dce371502c3d7de125430a5328dbccf7de88aaaf562",
    "sha256" + debug: "b9fe2ffb8a7a7f6c6c62844ed6e5d2da3de7b98c3431fce29d4f121032e928f4",
  ],
  "executorch": [
    "sha256": "d9dad164bbc9a371715c0798239fab2d1bfbd138c9bf99d7d7d8c76ad927ea27",
    "sha256" + debug: "cc6b6bd0dffd8e43177a556068832aef317de2a47cd5c1ef38470baf4262f4ae",
  ],
  "kernels_custom": [
    "sha256": "9bdc925355c3db8c1b68d8519b76dc9e9f3ce709a0058af254677763a08162af",
    "sha256" + debug: "a45c1eceb4e72519a10848ac51fbfc1d663c24a07528596efcc84abf90826450",
  ],
  "kernels_optimized": [
    "sha256": "2c727fdd298ba707efb298994d615fc73602e5cef07f06334a482f6bd5983290",
    "sha256" + debug: "2cd9d0195e1db9266dc0f3ac2085a8036613030933279c546b5ebf6c66241544",
  ],
  "kernels_portable": [
    "sha256": "064cb2b96c5d5fa0f1a5580b456ddb50e2d288ea39470ed416ab6810642eac76",
    "sha256" + debug: "5846b6d1c0cdec93396993eafd31c771bb714274c64f4ad3f070130db2601b96",
  ],
  "kernels_quantized": [
    "sha256": "52ad83ce471a36ab66413474215ae9d3a867e2d6779376eea8f18a8b937d0d1e",
    "sha256" + debug: "ccf04ae4964f4a9f07f5e609e0cd475151220d85f69ebe1578593d556f77bec6",
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
