// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.2.0"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "coreml_backend": [
    "sha256": "2c84a0ea7a38d47a50e2cf2fd28723e1a95222075a2960dc465aaf310112ea98",
    "sha256" + debug: "91b587d5271b54df0a2634dca7a860bd15b124b43fe1c6bf2a8550542be868d9",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "custom_backend": [
    "sha256": "b101b33ba77c7f47af089ee3ecc5d0d2bcc4567e30a4266989e2887a9ea0f4c9",
    "sha256" + debug: "564b78987fa39af743314a893d09238830c403b2906a022936c3408803a633b2",
  ],
  "executorch": [
    "sha256": "744a19b9463f2be781066031e39028b672408263b570e6be8348bf2efbdb019a",
    "sha256" + debug: "eff78cac783f5dbd2cd79a9173356e38cf222e0751169ee6795e17e555b14944",
  ],
  "mps_backend": [
    "sha256": "8788b792ce7d57cb2898841a92f50ce902a9bf9d87a56b9463f20a6a16ad8ce3",
    "sha256" + debug: "6a512910e4eda964001afc3ccdd80a6c7920bfe6427d7085f6f725480d713359",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "optimized_backend": [
    "sha256": "8db8de089a6e6579e9ae80551c397737a71161fdfa4588a8a2b97649c0ccedb8",
    "sha256" + debug: "213724b6e9f37d0c10358d86d5995daf53e68d51c1ee99b46e3288bb44d38c8b",
  ],
  "portable_backend": [
    "sha256": "f0eed2efa3aa21670b7e831f2ed8dc6b124f652eee7de2571159dce04b70b6ff",
    "sha256" + debug: "270c361b617870e35db84fa6a34ab250d75442466df466c5d3aefa0221d78b5b",
  ],
  "quantized_backend": [
    "sha256": "a2ab5effb9455849c0dbd0dfc5704ca1c1d32fd4a734be3ff3e578f5baa911f3",
    "sha256" + debug: "aaabce21c8352d2c1fec2598a1fec4c0359e1b606c5811f17f367831e754f23a",
  ],
  "xnnpack_backend": [
    "sha256": "ef7b9cdd653fd1094e3969b3bed1d5a9bde0f4ee751d70276a47954ae55220dd",
    "sha256" + debug: "2a1da274bd39c7278f4d61c240dfbfbaf480c2fc226c0da06f26d91f0623e0b2",
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
