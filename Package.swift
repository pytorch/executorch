// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250302"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "f9a8e5942a45840b16568326283ab79098b1e0284b9f739e6f249c3ca32244b1",
    "sha256" + debug: "4c41061942040b03731778c18b2661405ffd43c91e47e5d07feb69178d2127aa",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "bf53cc96c4ce5c7dee6bc1d44d1542bc026907ade8c58b21253dcccc760116bd",
    "sha256" + debug: "651ff830fbebef80360155c37d323959654769520d9eeffeda1d68d8323a37c6",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ad87f380a29e5a061d55020ea9297ba5cdd699cdefb4d6bdc79bc40635d27f21",
    "sha256" + debug: "ca9c3ac9c513f669abf0c71986d8ba56fe7d28bdd4f09ac3af7d8d919b5dc772",
  ],
  "executorch": [
    "sha256": "deedee3b44c461579abbe8ef23fc8fdaaad39b884e0ebfd662e992bbe5e850fe",
    "sha256" + debug: "92d309eed1c16c53fe94ad9f263ccaee7aeb9f36dfe5e9d0ac8a0c3e99dfea9d",
  ],
  "kernels_custom": [
    "sha256": "263838d071279710222d671523699f0f49436b9c9099ad8bfa960d2b973c4b52",
    "sha256" + debug: "a4db347a3e47791c837d0f0c79d61318504db9fa2c75f44d1a70f2f3aac5f9d7",
  ],
  "kernels_optimized": [
    "sha256": "b515f1e8a0bb0b829834fbc50f8495ef4a0c590f40adab61a2532022e5653f5d",
    "sha256" + debug: "0bc1e8620b4596a9217ddce759a751a3c984b77e56c99a695eabbcc05d601028",
  ],
  "kernels_portable": [
    "sha256": "17fb2f8549f1853461659fc30b43a1ecc0303549ff765e52991e4e7a97f60fff",
    "sha256" + debug: "323382561b520590a77312c605dc3752fb5362714c296f1edfe177b628f759ae",
  ],
  "kernels_quantized": [
    "sha256": "a124875bfb1abc3e3991579da411c6ebdd98552c9ffbe345c3236630864ed170",
    "sha256" + debug: "c31fa500fd36749d494cdc8260645f81ac3b6b6181dea15e9e985a02be4f591d",
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
