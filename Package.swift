// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250116"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "21c02e61b17f5cf6154e477117b6847708dfa6aeac2bac0eee93d8c79ceb1dab",
    "sha256" + debug: "92d3b08a3d1c1c291482232c4022e4cd1a68e9134d86f5c343ffcfd7d73481b0",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "443d0d947af04b06857b6f7ef420bf75bbc5fb441c8c3c0ba9cf970513aaa381",
    "sha256" + debug: "f7cb3fb4d48aa10c3fe22e9bfd12bb05b146471e015e717c556a4293ce996fb7",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "86e4e85c9b9d58fec43636f7ee3c857fd95e16607faa310047704bcb46ced7b9",
    "sha256" + debug: "c71510e95affba24f01ae6a2087f3bde78dd5c388bfb63b92e066c8e27f9b615",
  ],
  "executorch": [
    "sha256": "2bd0e9e2ef97bf5a909c1ddfba65582c62e03c13ff3f2a7ceb439c74036d3fc2",
    "sha256" + debug: "3c79142d42e3435b9628b0ddfb6e3c91785efcf68d3fd9df1728a5deb7dba3a1",
  ],
  "kernels_custom": [
    "sha256": "f73bad649dc90f0d73366981c9d39915fbf8304e17b840be91a485f1f2de4caf",
    "sha256" + debug: "5bf61c9a496b83d02843596f02d50a7f683e987f63b00d63acd6538906d52e11",
  ],
  "kernels_optimized": [
    "sha256": "e412d8a56055cb26e1c50dab536dd8dcced85f1a9ac323b54813addf6fcd3ed0",
    "sha256" + debug: "2abbbd343ce6fbcbfa9a07575bc15b3c24caf563c7562e61116c4c763e91e86a",
  ],
  "kernels_portable": [
    "sha256": "7e88af358f2018d45efab1d1429b86fc3c0aae294a38a5f749d07a925e406d47",
    "sha256" + debug: "8387a37f08b4e2894649205277b05abca87425dba052708e0478c2d2085054e4",
  ],
  "kernels_quantized": [
    "sha256": "c7d07298e4364f3018aa0648a7def983092a0e19ae2988f46206b23cf0a99862",
    "sha256" + debug: "e6f66f00d69d3546b3c8dcf3ec580f32041c73a47563e3eea0c780e52b7cf5c7",
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
