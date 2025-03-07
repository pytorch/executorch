// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250307"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "794025351bda307083a0af7d5427e34e581422783ab53bc46abe8b061132f19e",
    "sha256" + debug: "8a5c88b823e5e0d457d44220351f7d86a8a1c71b379f3778de3577720cd6384f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "9281f9aea1c8c9a6810b1f7b5f489918e7ee3b3e02b29f590e033ed670088fae",
    "sha256" + debug: "21e1884ed8375b4c38eb10783378949a7bd8b421e97ff39d4baaccbb1d308005",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7c73ed8f21659d39d3109f179fa7e4887240e2549b1a21c85dcfe181f2c2cb69",
    "sha256" + debug: "5913a21168deb788c9f9646c148897f41398ea0fd9d66236c33c1b9ad421a1ca",
  ],
  "executorch": [
    "sha256": "e14a6c64d137f6b6d093bd48c9c9a63915a3f23a35f6a3b1a2092c132e7f95b1",
    "sha256" + debug: "05272ec7a10760868a7c47a68eaf26a3b518dd75a0efe6dbfa555e0de234bc8a",
  ],
  "kernels_custom": [
    "sha256": "37d7de6041d550589ee61a52c4612408c1da4ac7899155a6f10bf50c09273be4",
    "sha256" + debug: "3830be545a41bcd20cc1487b23b0d01c581b51983280b8e5e42f9133e89e29ae",
  ],
  "kernels_optimized": [
    "sha256": "f686b452c6b69c96c697ed3928c3d668c31e7098a9cb652ba42e92a9711d0e5d",
    "sha256" + debug: "9f4905203c69d1aacd8636e5b3946f54460180d9e1ac86775fa538b9b0ecf980",
  ],
  "kernels_portable": [
    "sha256": "95550e242065cc7ffdbabcdda09033ab215e35f0ecc0309bf584a422ddebb8ee",
    "sha256" + debug: "833fed64fce0bb7711241185c9cb5f44821571f803e7a332c618109d24d24746",
  ],
  "kernels_quantized": [
    "sha256": "fff992a8df2d4b169377edb0de8b782549bc5c2781120919fe82bb2afb07a869",
    "sha256" + debug: "905187237e5fc6952579f51f1ba211254c6d52eb8a7084d552d82baa53248f7d",
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
