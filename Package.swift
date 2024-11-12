// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "latest"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "6a4e521e6ab480b92bc13b1bff41069cd75d25e5752a035b04b076f6e7b09396",
    "sha256" + debug: "a08025249731874bcda1704a4add13a15088068529a3e3d0d47d85ec0043c182",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "9183da3e0eadb42fee140938d70f218c63cbd269a3ad451142ab18a37f34769c",
    "sha256" + debug: "d886805a07a0587e9e2bdacef6da69b31251039a66db8cda8105150e0434d5fe",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3722f24c4f9796e9bb95db6c0445d808a8f5c64a6fb51b319bca502c64f32ff3",
    "sha256" + debug: "027c75f3c668221a2a11d7622b08d17b52a34c47a7de1fb80a1a1ea008cc746e",
  ],
  "executorch": [
    "sha256": "4b623a97b325eb48bcdae5359929e570ed5308a09669b5461211a7d693850579",
    "sha256" + debug: "97773632fce699d934ddebbc8359f775440565cca9a338ab5ca753e4bfe87d4f",
  ],
  "kernels_custom": [
    "sha256": "821f66ab0171f35df2ca011e246ac9926dff171fd51715c1a836cd13bf156d3e",
    "sha256" + debug: "cc0cdf14e9b538c8059430e25a81e36e00ec06bf34a32c41484907756a9d0c8c",
  ],
  "kernels_optimized": [
    "sha256": "d53018beef442f1c00f3a025bf12af37ef67e4df8ecc349e71016321b7d9dfb7",
    "sha256" + debug: "9ad9aa87457d8588d6ce1191b2e28de81e785fd4d04f1026916816592a1d1004",
  ],
  "kernels_portable": [
    "sha256": "7d709843c802185ef0072ab79fa1f2a76fc7432ee64b26ddad09f1cbcd1dc85b",
    "sha256" + debug: "240d0748b240af3fcd207b6041df4a343c5c811d6fd3d3785dd09e09b4d7359d",
  ],
  "kernels_quantized": [
    "sha256": "dbc43ae67a92d0953e5e9722876af3440168e3c182945e5ce323e30defa0169f",
    "sha256" + debug: "bebd23bd351794e61630b4dd80943c9c97bd7b998fd7afc49930096259aa8bf4",
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
