// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241206"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "3707a0fbf5c5dd14bacd0db6b20693e9f26dcada7e52177007fe0239bb8188d4",
    "sha256" + debug: "09b3f58502f34d20d5fd19a251e895f430f59b3765ecc1d179b00e722f0ceb17",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8203d5d4bb0d7447f1f684ddee4ac7b130174c48210b0471d758f0fe547784a3",
    "sha256" + debug: "dc8412c47d3431add4f232c7f83e82e32105ca04e9438181ba41818df0026e25",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b67e41044d086cc0bbf6abc9d1c504ae16445dce616e05b337d10741e999b601",
    "sha256" + debug: "4333581c9a0b458525ca6fda6d9f4ba1644b7bf3a5694f8e1ca95c3fa65caa19",
  ],
  "executorch": [
    "sha256": "c84325d351790aca4d09cd6bd51002886f603b9bae07807d03f67e57cc2a6569",
    "sha256" + debug: "c82b1aa3269b531264aa5662d1c2fe71cd9ca384475f9a742a7461f94fcdfec4",
  ],
  "kernels_custom": [
    "sha256": "a0192017dbd6d10a02048c94c43c84e82eea67830d11128f5f8498430abecb64",
    "sha256" + debug: "d3a197e524d087e0b70871577d0f02114c13e3c22562b4c39420e2e12f99b6ef",
  ],
  "kernels_optimized": [
    "sha256": "5f0bbdeebc8b9fe21ce3056317f33cd8138942af42bba1331e43f956fb256e53",
    "sha256" + debug: "b1aeae565071bf3804e50c241452ecb9160cc7f3b5b607d2c0350d9316ad05da",
  ],
  "kernels_portable": [
    "sha256": "52c6f1ac528ed19f7940ed3e08b0b4dc691ce934e7a2e4654b6bcc826c5d180f",
    "sha256" + debug: "816cf73f2387289d63f14354cfef286eed33932d8fc7d680d4843a3f41b0d745",
  ],
  "kernels_quantized": [
    "sha256": "bb2e0a62bce2f4e4ebde5283c4d1d3e867d156739794085912c03e9916a6da3c",
    "sha256" + debug: "2cbb85db272fd4cc1e34fd2c1aa29a2ff2a5082858154fe8ff724a9561b77a39",
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
