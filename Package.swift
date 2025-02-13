// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250213"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "761a2cca961fdbbea2397ccc0cb9d1bc853446a91c9f3980ce62dd3c20ff0061",
    "sha256" + debug: "c9c74c555571a3513ce97fa60db9ac1f5d7c35874adc49154ac214991e96472b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2cbea1c6630e2b068b36be2ff06fc748c8d4db85de2f15298c9c3435b9dc5a00",
    "sha256" + debug: "8170235e086e218f2a7396e4a1500acd31ea626f176b89a29bce9f104e0d1a7f",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ddf7992fd5ea67df4032f9577b386556af01632978c13a34cf4c59f31baddf08",
    "sha256" + debug: "3dceeaf4376c43a7a0e65ded72b5d3fcf561aab19f94928c0447430cabb11d82",
  ],
  "executorch": [
    "sha256": "88d5fad25cacf7ebd5a9be81aa47118d92cc3e93a71533198de19e9e76c78d1f",
    "sha256" + debug: "d4a6501a103f9caacf128e7554724afc513898bf24722325c44096572abd1d34",
  ],
  "kernels_custom": [
    "sha256": "6b38a37fb1f1db00f4d4ba268b08a95fdcd5dbd53de6d7f81149598bab2133b1",
    "sha256" + debug: "251baea6fad44ad8983b897d729a2ef9d138a01b666436fee68af48fb7fa0f3b",
  ],
  "kernels_optimized": [
    "sha256": "bda38f2610bd3ea4ea90f1868e84be5ffbd8420562d914fdb4bde4fd993c552c",
    "sha256" + debug: "be5851eecb61c1009cceb628f1980e565dd64515dc5821d8026137ddfcce889d",
  ],
  "kernels_portable": [
    "sha256": "c67836b472d7112ae179ec8bfa5d088b95967c3f9f9c3d1b90da64bddaf7bfb9",
    "sha256" + debug: "c1e83948c9577d19bdf03ae27e7ed9958ceb646ae69b8228ce9c058fec201122",
  ],
  "kernels_quantized": [
    "sha256": "a6c89137da5d67ed59b5c6d3672523745e9923b0d8ede93d3bc6800340a09e9c",
    "sha256" + debug: "021e55d4b9c515ddbf1c374a590edcffe2fc7187454c5f2e7bd15df1ea24925d",
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
