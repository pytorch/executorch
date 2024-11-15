// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241114"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "58c55b5242e35f8141eea0e0ae87d332279286e1887cf235858ff7a29f4cbff6",
    "sha256" + debug: "c97c176d399a7551227175e57afdbf0476c00c4c24a69827db0ea16cdc17c5f2",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "da3fdb3bbbb0a63728fa0314e94a718ca97943292df62c8583e9c3778a7b083e",
    "sha256" + debug: "41714f2c7dafb975b61ac3124fdfae71903036d4d5aa4345f56cfcac6276f177",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ce8e8595776d985c89ea264d22abb5d1003b076e0c0fb50a7df3e67088561d30",
    "sha256" + debug: "ab0b81a0bf2874de1cfb42aa37e7ba4751d5039a8a7ca221639f91339439ab7f",
  ],
  "executorch": [
    "sha256": "f6f027875f5155a1ddfe6c205d689bb41cf0c554c7812e43255b0f59b8fdb9ec",
    "sha256" + debug: "efda23f7bc63dc6c07fccf8003f7d6100f29c43454cc7aa1ca77d3db59e8e101",
  ],
  "kernels_custom": [
    "sha256": "d0b6a962da4e191cf2a3da197484107d437560381d3de50aaa854ed7fff1914e",
    "sha256" + debug: "d7946e7866dc9346da2862cfe7f7bd230a14c7259f93b2ab8012742160631b62",
  ],
  "kernels_optimized": [
    "sha256": "8165945d49ece11d30c1082125c81105bc6325d6097177c5651aef577c39c4c5",
    "sha256" + debug: "9af4d26b107f6e6db8c1cc72548b62e3c4986c423713cfe6e6133108630cfc5c",
  ],
  "kernels_portable": [
    "sha256": "dd2cc7e5a5ea1538f3b39e369163b1e40d73e09f82ff7fd593592ddf5edc7368",
    "sha256" + debug: "7d123a23f2b5cd8469c5fcd55efb6ed6f11c27480822f405aea63ea68a8f1fbe",
  ],
  "kernels_quantized": [
    "sha256": "f2eaeb66f7fd1a1c1b80ca9a0c8fdfec58d38f17a64b46a995ca38127e16c00c",
    "sha256" + debug: "e8ed2eae51760b24b7f5bfa94233f2dbd47f01b0144e322c0b508e9c23dee59d",
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
