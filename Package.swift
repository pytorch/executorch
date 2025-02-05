// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250205"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "f1fdf7782d2ac7de2f775d19352b8f09428cee5afa0fa5b59c8e60df3369d814",
    "sha256" + debug: "06eff8ba620da7858c7aac313d3a74e017f72be684a1ed6ddebf06f11e6f5c4c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "904d91f4604e6bab8cb57a16c40b9c4f1774f84e93bd64500d0f2105a8c65960",
    "sha256" + debug: "41e82cee4b50492ff8fe00941931a8a4e639e6772cef32c947be1b5dd1c9ee68",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7cd652089bf32d7e5ac2ad5300bd94cac9bff21fb506ec15dadfedd606809ae6",
    "sha256" + debug: "5f6b3a65cae773fe7f694652d364b79dd333614d91a2e9b24d3e3580f4d785d8",
  ],
  "executorch": [
    "sha256": "2265324ffafdc762ea64303d3acf4454a739a26d0ace55e4c2c565e28d8df3d3",
    "sha256" + debug: "dff20a72dcc185504a16530e88d1dc48dd7a7f7d16a71f7fed6cb0b5511f3169",
  ],
  "kernels_custom": [
    "sha256": "99ddc29f7e5983d3a9a54780843d9b0147933e359e17d7a5aa7d3c23f8ed8718",
    "sha256" + debug: "7b29ec27386c3987740c855e17dfe8ea90e32c19ab405ae6778b8dee8cd8322f",
  ],
  "kernels_optimized": [
    "sha256": "2f1432b6c8ca458e73b393d11ab9a04f8d12a3a73e362d174966644e8ebc7337",
    "sha256" + debug: "bd15f55bc5c39b48e502ab0917a890c1fc6709ced4348c23d94af2e2bcddcf33",
  ],
  "kernels_portable": [
    "sha256": "87fd1866df008f88150c635a5a90def6311a0cbdf71636272fe5587d543a0066",
    "sha256" + debug: "1591cb4301cb6133bfbd545c5f9f0ea15c7edb5812620a95663e0b50dde30086",
  ],
  "kernels_quantized": [
    "sha256": "14b561cd2964a95464978b426f0d2e665fb0c12697707c1df87d2a7793286bc9",
    "sha256" + debug: "6aac230694aa2b671875c6a4f73682017c4c247bd8e4fffe77fa38dd00b1d30a",
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
