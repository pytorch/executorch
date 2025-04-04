// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250404"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "3e26da5c44c3389f82a7b3e955b5f67d95d7933da776882bdb64adc486529762",
    "sha256" + debug: "81349586240ae550a6eb4219d80af3356229bc1c18b02b245e7f91b76e57fda9",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "f84aa3cdc08f8a84d8d0e9b3deb2d4a2047402f3efd8429cf4961067e1b4b1dc",
    "sha256" + debug: "0361fc0f4625ad092de820596b9f4366763b4f11cc7e6cdb3a2268781ee26d54",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c046aca5f7c64e46715ed5405a9f1ee32f89d83dae811ecc3c78ab039efdf784",
    "sha256" + debug: "665c1beccc2dc1a28e5406368f927aa17d4caeed201cd54a9ff57842a99c65f0",
  ],
  "executorch": [
    "sha256": "410078b427f822c4449a606246bbe1831c780fbf8d73b6a6213ed00f468c2a80",
    "sha256" + debug: "316fe3372f2925671dbd9a6d5e3a66fae622f5d31e320ff39d5041bbecba9f8e",
  ],
  "kernels_custom": [
    "sha256": "2c079c2ab4320b84a63e3ebdb5ce21f038b807b54166dd738758978fe61a5d30",
    "sha256" + debug: "03c59a6c4f47f591dac97d18cae97bbcca3c40b453d41562d8f111e9a8ca7ac8",
  ],
  "kernels_optimized": [
    "sha256": "a2197ce9035949e19335da36a098f76dae6444fd612f032f0cdc86071b653147",
    "sha256" + debug: "ff9e8dc1d05c19b6e383cfa8fb4039b6a6ca3a5c094bed9b093af02e9ea634e4",
  ],
  "kernels_portable": [
    "sha256": "092655c4c418ea90261086e2ea1dbc6d4e4a073e7f38ab89d90e1d4e97c79981",
    "sha256" + debug: "9c1ff4671d6e2d5fe84e4a1ff9b13f6f645e62dd19de691dad909df5d312d7c7",
  ],
  "kernels_quantized": [
    "sha256": "641d9230a8a2f183b902eb9832e750074a16ce94dc14493aedd91541895c3725",
    "sha256" + debug: "aac38832b41a27181766f6284a1361065b0de1fcf02f09263abf46cd9f94269d",
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
