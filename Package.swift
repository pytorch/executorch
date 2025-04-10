// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250410"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "7659730a069218d00dbba3922b429497904046c40192d1fc114d915d99a6c7ee",
    "sha256" + debug: "385afeb25d3d18298a91087c9048aadf8f8e3ce460e63c8f6f0a472ecc61d3d1",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "dbc3764a1434ee34003c74a7d4fa3f6d1c251cd5820ff5bed8b4a48885d80d59",
    "sha256" + debug: "e91e220d367f5f4dff1efe9d9c06ec1bdb0339906237da9139bc05320de6159c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e2221035a13b674da302338da8fab8ea97a62b3cee71d12890ea339722a32313",
    "sha256" + debug: "b4fa37b2471ac19d3095d1387c2eb44817784403e852c292276f22cc0402afab",
  ],
  "executorch": [
    "sha256": "c66545d61dfc03457bf80e8987c6174001bbbb8346086e4f06d415ee010f1038",
    "sha256" + debug: "7a1f3e1b81cc2ad973e6372c1546d7415ba6e5928deefe5aab84c916a230c7ad",
  ],
  "kernels_custom": [
    "sha256": "b8da7d2fbe88873db6b2a1a3931ec4fd6d6d18cfcd3983098ff53b3eacdf4e63",
    "sha256" + debug: "e8d455fea36e94d27b807fdee4cd288907bc6389cc014384ef9fd1589070f137",
  ],
  "kernels_optimized": [
    "sha256": "b9b8d09c429686b431067ccceaec90685075b36df0c3ca855162a8dc0c6bfd85",
    "sha256" + debug: "e2b3b9dedb8eeb3e582d7a5ba350fb1a87065ea50488577bacbee9b4407b7b32",
  ],
  "kernels_portable": [
    "sha256": "d9d6df11c4af234c429fd1f353aee784e584b42587bad9f82fb1b0175dc49258",
    "sha256" + debug: "f9fe177ba14b20cff737562d9f7e0dec3fe5f2a4e242356cf02000447e064323",
  ],
  "kernels_quantized": [
    "sha256": "2aec259579ccce6105409c08eff64751d9cca77adcb628296d37e6ba83a203e3",
    "sha256" + debug: "d6f17f89f15326af9e7679bb5be43fa0610286ecc930a1ee5a078c70f775a898",
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
