// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250109"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "0f169053c61f87897703ccbbf81713ce01f285a22fc585b7a59184b2053ee3e0",
    "sha256" + debug: "876e08d5ed5a9652aaf987bd9b50940b358c259f067290f1f835ce5a01e60efe",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2aa33d049fedeed8347d2bf2f51f359119a6854ce4db772b97b9ef2476bd7e26",
    "sha256" + debug: "1728316b707f211a5993a34d4edd094fd61cfc050307f75921db27b9237efba5",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "252fc6bc33f377455dde00cfe8130338f000718e22ec32a59c2c5ecbe4c0a580",
    "sha256" + debug: "906305be95587292077acab0f66320f6b639e73b9b8115b9cb80b940dbfbe885",
  ],
  "executorch": [
    "sha256": "ce624a363dd1f70bbb0366db7c381bea8a82efbf25d496930e60689d74ceb949",
    "sha256" + debug: "bfe20d4264d3454ee0f15da60781e72174dd76648ec235de3e695da53a6cf8e0",
  ],
  "kernels_custom": [
    "sha256": "02fc9d628c832b4a9d5ad987d35f73f8538fbefcc72c7b33201bc9196c02dc6e",
    "sha256" + debug: "1cb2788ce600676572c7e13da42a1a44eb3240da59eb5118168354d218fb1448",
  ],
  "kernels_optimized": [
    "sha256": "3d878462362199f0d5db248ae0f9c0eca7ccd03d5e06847216a53332cf0c65e9",
    "sha256" + debug: "cd42383dcae965c066939566a46725baf5674cfa5c8e6f7b5323c82eb366c4e6",
  ],
  "kernels_portable": [
    "sha256": "20830f5b0e84b66dd99bde7bd93eb8357460cd18c3689cd05ef3dd3c7290999c",
    "sha256" + debug: "488376c34247fcd4a89178ba9810b1749be6ae65c3fc2be8fb75d5a05cc99c65",
  ],
  "kernels_quantized": [
    "sha256": "ba55a5858750e714f916578c601054fccad45edfded449ce6dad08671c22cd6f",
    "sha256" + debug: "56d85065df8830902e5a87b6dc5eeb55addaa70a5761c8d69c2eb6d61e528192",
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
