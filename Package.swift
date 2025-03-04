// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250304"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "f353cb93dccc31153625329bea03b429ce4d0c5aa302d469450820ca44852ebb",
    "sha256" + debug: "dd5a8c7f1371124c04a6e2f27758e1ff5f0bb81ee302f1d02042bf0d8d5842aa",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1d9dc8df1a3253fdfae17f5b604d2243ba1c48cb0e807d17efe64a9f3af35da5",
    "sha256" + debug: "271a14c2e100d8a4614f1773dccd229d69e92aff76421261a993df216a8e7fd9",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "fb3644853fb69ce80d587a8888e8ec1ac819cefc1cf5431758e2ac9072fbbef3",
    "sha256" + debug: "f1e873b319388cf87b1c8411d2e6e5308db142d06c75caaf119e05a01d80f9d8",
  ],
  "executorch": [
    "sha256": "6b8f48158596c2d7272d08617fcb10ef7f35db8f19bc5c716fa5e7718ee9c3ae",
    "sha256" + debug: "6b8bcfe3d1f98f39d65e608e1980d18e8f326df10be8b57ff1ffe3e894485780",
  ],
  "kernels_custom": [
    "sha256": "cef3969c270a9f166604a72ff57b1fd588f366e1ea608c51f18dd36d3d2af7a6",
    "sha256" + debug: "e0c929486d30898f85f33f145964c259a9775e4b444ac58e7ad3e531ea9a2854",
  ],
  "kernels_optimized": [
    "sha256": "10c856e9a06c0b9e58b7845c536acd23c30a4e9510726c2a3afc55307153713d",
    "sha256" + debug: "dfddad62e5af37f7373b7c7f4b50b7e6fff4e4004c31e12e2b8b669e6102209a",
  ],
  "kernels_portable": [
    "sha256": "34cce18ad8bc2bfa19b82488e39dd6d46c00d28b821a10944e2d156d34509e83",
    "sha256" + debug: "a22e9366a64b3786c934793cd99d148f7a9139c977637415c3096996d53c4731",
  ],
  "kernels_quantized": [
    "sha256": "eb9e2a1151c281e43b6c7b6c1476ea452bbec8f8d217cb5631b868a0aa911848",
    "sha256" + debug: "d39500ece17d669e04b68797985b4c1b01e774468d8ad2c49d2d6dca6c8fccb8",
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
