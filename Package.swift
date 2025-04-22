// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250422"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "2173f60918868d7620ef83c4c10ad1ed91041fca2a274ed3c06fd86cb7df88f9",
    "sha256" + debug: "e237bb8508561e29d8805f85706666e16e6fb92b9e0e1927bff82cc31ab429ec",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "4bdd720897fab447b49bf899d709e649b16e69f6b854a1b4af1c9839e0150bf1",
    "sha256" + debug: "1a9d022f5c5127bab19902d38c108b22c78951af018bb4e7d4a6aeb9204f0bba",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f4bd7607d656433cbb5337c7893401bbe2a1bbf2fbfc200a3a6104dcae9ee01e",
    "sha256" + debug: "459502c47267536fab8564eb65d046d08c012121ccc3787049df80b9bdde675d",
  ],
  "executorch": [
    "sha256": "eaf93995d89230de0bbabad0a8ed67b3d61d4aae60dae04c8cd40014778a263c",
    "sha256" + debug: "743df53f8479bfe6f7d50eb2d6d27b7ce88d4b2622b0d1b1a86db6c676bfc4bb",
  ],
  "kernels_custom": [
    "sha256": "6f96e9772c8d2d12839c9fa356e8bf7fe4a593688c99199d1e7a21972613066b",
    "sha256" + debug: "b893efbf67659d3e41e569b3a5771ff7cbbd41a610fa1f6a55374837705e6df2",
  ],
  "kernels_optimized": [
    "sha256": "c12c32b8295d1789237cb8235c675b8a4ba0deba5153b6f2d13b2e268188294a",
    "sha256" + debug: "8f122a6c4189ff51badeb961d6ed37ecac4c7c2eab2080c0bcec5b0e3c3a848e",
  ],
  "kernels_portable": [
    "sha256": "e0b68260f5720da1df7e7b963c1ef0e490a0fbf2268c71e614bea4a7a95b4670",
    "sha256" + debug: "874fc69395aa8aa98e7dc7809099e055581fcc647deeac497d0c0700ddf7fb90",
  ],
  "kernels_quantized": [
    "sha256": "bf79cc205293f2a986ae4a1a68456dceea368adb786ca6a0a8e79598f2c2102f",
    "sha256" + debug: "5ecc0a784fd0dcc8a1bf1d30860b87665d80f68faa5b5063a62e91190b9c5d56",
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
