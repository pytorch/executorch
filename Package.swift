// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250104"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "b2f6ff8cc7eea8cdc44d269ac55dc100d2c95543e22a55aa7dbed8621ae5d8eb",
    "sha256" + debug: "6beac3778229e8bf06f48595628cd551a5ab513133502ae5a5381e3c6f56073a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b5e3b594d74a8dac9481d3c9f2691a50ae6ca20be43cf16279b74dd5c96654dd",
    "sha256" + debug: "0725b97c4f0976f788fa91782b1d9a7f3e0b8713aadbb24a274ccb61042068ef",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "07a0371caa0799f3d3338577af3bbcfa5871e564f5c5a6130280efe0d47209dc",
    "sha256" + debug: "fbab7e9cca5e9abbac9dd873bed8ea92ea0501dcba36c7f4a22e26965c75a026",
  ],
  "executorch": [
    "sha256": "d8d40429fc4019302e7c3a5ffe4ca919d87c082ca35cdeea0e5a0496c10e4b94",
    "sha256" + debug: "734e4af6fc5a5f3c5a9b6dd181f3902181238dddcc994e2462b0efdfbd03cab4",
  ],
  "kernels_custom": [
    "sha256": "bfedc355e8c8ac6de1ec8f4332323d9f3b85760fae169b3dbc2494b018336ebd",
    "sha256" + debug: "db4fb79688e998af08496fc94a0a36993ed0dda0e33be6407b2adaee7f03e023",
  ],
  "kernels_optimized": [
    "sha256": "c6de5c4567e5a8247866a9b0d150423e2174c70ca9e2ca09d179ed0b1ac37699",
    "sha256" + debug: "3c072f587ec2fdb0bc9f1a42c9ce063b5c10041c3f1411614bf7e4c76de05b25",
  ],
  "kernels_portable": [
    "sha256": "35c1ce059b45766d300f4af41179795b876cf239ecc9d960f6b74a80d5cca6e0",
    "sha256" + debug: "89b65dc87155b05d4020502556636d48f96640890407452639c25f7d74d7ed45",
  ],
  "kernels_quantized": [
    "sha256": "21567f82248f2fb8d339b5ffc5007ca073e214428e37bc2fe8789c632d845c3a",
    "sha256" + debug: "f4991bc385598a87d31c7e41fa5239d009b5ab83669e4848b97932c503f11826",
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
