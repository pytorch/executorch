// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250416"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "c065888b9c124fb482950ace909a6e73460b2191b9962293eace46c451656e81",
    "sha256" + debug: "ea244400648f5e4e2827348ffe5f40c562e42748b855df4b132a57e5deb6ea7f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "252c84e035cc2d7db8ef4e08d092d8d7dfc19f758bdc0530eb1d520a75efd297",
    "sha256" + debug: "c911943edfde642d55e0b6bc741494127c0f82ce47936da361d38b5ed04da24e",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e5d5bc9c4dbc5e8d6f58c0f523ca7c9c217ee8a81e8494bd128aa85f2f4dadad",
    "sha256" + debug: "8ba070b88cf76d6bd7ce7cbf7b8a132532553c83364b92f6026225131ad3b1b6",
  ],
  "executorch": [
    "sha256": "2e04541733da71d1d5e409caacc101dbc84b1d3f90eb6dd7031af0c0f4756396",
    "sha256" + debug: "e9772f6b15ef1f93cad3920be95d7faff1a5c9854803c15d8bad48e4c2b7eb9b",
  ],
  "kernels_custom": [
    "sha256": "b9fd532be8f7c87f5c115fc9ca1746e0724428fa902453ea8b99871cf338f7d6",
    "sha256" + debug: "b40c8987717855634933071ca69d191eff00a86872030d889bcfbbe6ff34aa9c",
  ],
  "kernels_optimized": [
    "sha256": "3b024723087f1e4f3180f8795dedf385ac7f8f02efa9dbb48c40d0edc948fdf6",
    "sha256" + debug: "3a4a20384f99b40245349a08f4ca28409088d946b4b7bf711d444115db714888",
  ],
  "kernels_portable": [
    "sha256": "8525b584e2293b288774a89bf6a9eee5f655e9135d1bf6d437d0cd8e19527296",
    "sha256" + debug: "40877c75a2b777ae751dbf5e823be0eff5cf026b5110bafd9bf5a9740996ec3d",
  ],
  "kernels_quantized": [
    "sha256": "fd4dd790956cadc2cb5a5eee120125126ea94c2ed8de71d28b9e7916d106a8aa",
    "sha256" + debug: "667abc3922e5f91eca588c3c8c6c9137c2ee7952c4281e8381569d16186131f9",
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
