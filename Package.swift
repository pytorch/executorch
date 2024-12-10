// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241210"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "070a0842ff99b9fda97b5ce608408a4d0b5e5363ceaa777b21e54420b7295de8",
    "sha256" + debug: "e43ff51f117fd30902ca9b220eb700ca82adefc13b48ee2bc0dcd7b63a0f2e7b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5430a814fccb5e4134258641f993d85d0ac78b3039789605a18048a6c8058972",
    "sha256" + debug: "76deeefdf14578f4f36f336112a375d53a679e6ee22f5ca08f4150471e7f12af",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f07b7732fd590481fc28595c1dd7d7e0f53477c3384c729852e33aa19ff25492",
    "sha256" + debug: "f9a24b505ef67f70bf1096dd8d0bf28ac7c8444e4a2cd5185eb8124c8365aa63",
  ],
  "executorch": [
    "sha256": "ea62b0103e19f13ddd81788ee9e0b791aee5898622bf9b83f32d976e514e9196",
    "sha256" + debug: "1ec59e9bb8e08895fc399970c20f53c2ce348552fda5c7d2960bd8d59c1bd3bb",
  ],
  "kernels_custom": [
    "sha256": "80e766b9300f208fc46dcfbfff801885d2d4da60eb8e78bb1974259d83e22c29",
    "sha256" + debug: "c70df062006d9f5769e6c03e9ca626030ba96057c730b98fa8fc23af3f2c4939",
  ],
  "kernels_optimized": [
    "sha256": "9515498246f92568e3a6a860080d45b8fa4eb54c5695fa80f4daeb0407f6b9c0",
    "sha256" + debug: "e74804ee932fa09f5f6b43e634552bf897144e001feed4d4c61ab35cd7b63aaa",
  ],
  "kernels_portable": [
    "sha256": "b362f27fbbeb6db2b0414699fb7f5e0084328be21475f01dbec956724915fe69",
    "sha256" + debug: "9604e6833474cdbe8f5b0bf9c52e921ba03921917d853bff8221fb38f588be2c",
  ],
  "kernels_quantized": [
    "sha256": "556b71d5ed1a350d39bd531b6ef23ebc7baecfa4a67abc26e4b8113c9f2216da",
    "sha256" + debug: "7f9084aadd649b1e75a39ccbf94744a94ca183fe30b7f521ab3382de29d0502d",
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
