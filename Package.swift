// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250326"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "9b3636d4eb0abbc9df89585c847ec7e5ab06983f98b3cb32bd746bbdc9e5b4b1",
    "sha256" + debug: "eeb14423167bd57eb4aec75a4546d35506b0bd4f137df967c0490a9306873577",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "40b932f362f484ba8a741361b61bfdd48f2aa4fbd3811ad9cd2344e8dea4a1eb",
    "sha256" + debug: "ec6b91306225a6eb1dac6118fcf0264c0a5060352dae58bf9e20ca3407df97dc",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "fe9947d296760e447ee869f83dc81c34b3d4394e1769349f7777a442a0d09782",
    "sha256" + debug: "42ae9f87daa0c1be35dd01493a6342f9a31dc547918e69514d11893c8bee4835",
  ],
  "executorch": [
    "sha256": "db552d8beff426683a5a13913ed01252201795d6ac5be39bf926348af0fd7d4a",
    "sha256" + debug: "39a80e1022f4837e65af5b6903edd854eda59d3c9dfbf43b14f1d147904517be",
  ],
  "kernels_custom": [
    "sha256": "aabdf43da7e697e9f9b445bfcb7ed2bf0b5cb3df11db52879979b156472a64d3",
    "sha256" + debug: "f0a41c68af00c2c5c249c83af4e7c07c0f4ce812521987e2e07ffa6956e46bed",
  ],
  "kernels_optimized": [
    "sha256": "944cbbf46311db6e458f791d83bdb804f4befc610a3dd3c3e17caf2975e1db6c",
    "sha256" + debug: "b6a77805ce3fdc7bb2a0da888a17087588431b2e83940f8ba15b688cee29fe17",
  ],
  "kernels_portable": [
    "sha256": "62d38a00f69bb21d1fa281ae0d23a409b1d71b39802ea18f74c9d34ecd1a849a",
    "sha256" + debug: "f8dd994cf552a7cfe129481d08b880716fc234490ca6c7735217877cbec45466",
  ],
  "kernels_quantized": [
    "sha256": "3c6208538d2bc8bbb759b58d5f7d87ab08b1d0e4c363925f213f0438ccf7643a",
    "sha256" + debug: "44b5c6f4eee60c82952da47f40a1bd1315f8740034f682fbd265f467bffd0fa2",
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
