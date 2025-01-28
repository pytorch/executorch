// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250128"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "49c9d5df278909c36f0ecb0b0155e32c3240b8ba0dfc9b1bff2f211f607fb096",
    "sha256" + debug: "84638728a56bc63550fb0885dfc25e42334a4930c1771f86d34e736727d77dba",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "950ea38bba76e2cdd6bb940e3fbff78fbc8f638594545d12fe34cc6102260d92",
    "sha256" + debug: "4002adb2251a333a101a9959e1529474f0e9a1de2b4afdd2665a3fe736057b99",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "76fd4b0322561f087106460c3907b8dcabb23821b8e28595a5676f8ab1f6eb3c",
    "sha256" + debug: "678205259720d2a0bde669f391c31459619a7102574f390ce0bc69ff0b411526",
  ],
  "executorch": [
    "sha256": "693dfe4ed6e12e7976f6f04c9e41e64cbaf5340bf42c1467fc39eb25e1806022",
    "sha256" + debug: "ab8fbad35b3b075177b7903c65d973c2546c000a2518dfe15b5c0bb6e092db61",
  ],
  "kernels_custom": [
    "sha256": "eaaa8a353c4cd58b7e10e0ba3d7a1adf9f9a54a5928159050e870fabafe51b46",
    "sha256" + debug: "9a4f7f4c4f76b6dd5c86bc4c8fa2a9436ef1f0f7e51b4900692ee30bcb562fb8",
  ],
  "kernels_optimized": [
    "sha256": "c985a5e4d287ce19749721b75e844047befd1deb2cf489e2b9b71d6d9d618e74",
    "sha256" + debug: "369002a50164f40b8b182db94f309c6f889f68864fa21cfe751c3f59ba20b024",
  ],
  "kernels_portable": [
    "sha256": "87868ff07713c45c5ebddf6a81b0e670f85562cb9506d3bdff9f3d6b3b91040e",
    "sha256" + debug: "7564767d34df4e768a8a180f83e8e18517eae5b9fedf7639615bfc16df5b4c5d",
  ],
  "kernels_quantized": [
    "sha256": "0d4ca1b4ce2fc549c63cadee4f29baf810e1a2fa4ac5971f1f0e15fdfb3fb159",
    "sha256" + debug: "3b3e8a34374747dcc7dcbfd8dccda1b518c21786b42ac39d27abab84b4d4f82b",
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
