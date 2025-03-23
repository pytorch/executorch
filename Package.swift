// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250323"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "c0e83cf7bedc26d7e3799fd1c0bc9380cd8d71ec6f7fcb32a15bbea0df281096",
    "sha256" + debug: "271c918d12fec904307f58dd5e4c2afeef249902ded59e2a0eec8b0ae3a4fdf5",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "edccb83b77319349eb5fed8f1cc104871accc10c07275427f50f6c9a699ed2fd",
    "sha256" + debug: "ff3a7691d8354087480c3790decaf9f02a0a6568cf5168dcab2192880e3fe2f1",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "66dbf120fc69156383afb8ed00b92a7d203e0e7f918519dc61f3eda5a0e63795",
    "sha256" + debug: "18370f6776f0f10ed2164ed2e27e7abc3ddf8871a01a73dcdbc8af8ca3461654",
  ],
  "executorch": [
    "sha256": "dbf101c36b5b81a23401196a76e0dc73ce9b399f192d79a04ca424716400d014",
    "sha256" + debug: "57cd6f107ffe750ea7b113189c60e5c4bf68c6b02ecced416d86880135f58be7",
  ],
  "kernels_custom": [
    "sha256": "55c138d98f12addca54fc9ce8f7a834567e9f5861b397226a045f9fdf0d604cd",
    "sha256" + debug: "c523ae5ff6f42b8aa857d646ec1c6ae518c63657e08bce5e584a7ce793a30d76",
  ],
  "kernels_optimized": [
    "sha256": "22ff26c7f215fdca34f017d47a4ec075573dfd4d0c8ea6a181df62b1ae79ab4c",
    "sha256" + debug: "0412b09a4ec58b74789887800e806cec0dbb6f8beebfb3530330de056a510591",
  ],
  "kernels_portable": [
    "sha256": "dfe61972784091cc5525f553128f004f1d5ed8a2af48e2ccd819ce254c923883",
    "sha256" + debug: "cf7b58902376c3c56e7f897d569a64c73590cdc2107e883c1dae9fc6a8b52b06",
  ],
  "kernels_quantized": [
    "sha256": "72d6f20c928551ab73fa70c7b0283f979cc3f1b82a61ee07778dda9cd3100402",
    "sha256" + debug: "6f21b2dc6b5740d55c4f31acefff91e0e6ed3c69355fd52f3495de3c3baa3e83",
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
