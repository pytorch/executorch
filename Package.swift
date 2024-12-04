// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241204"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "cc91fb505a892478ecfc6369818407d1000df421e9921b7fb7dd5e54cbde8c7c",
    "sha256" + debug: "45bdf6176eec3cc11eb507be672d9de4bb6f4d8a715743b3ba44969fe5a9481d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2e26cbba4f3a94695c3555fcfa17b29ebf7cc49f776ede9d59029b2e990f45bb",
    "sha256" + debug: "3a174597b48299d47a54c3e9a7b6876d259518ad5287cfa18812a4f2ffeb1d13",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1c55716a7b4bda83a70b73ab060c26f34e5bb083bdda383cf5153694c3885287",
    "sha256" + debug: "f62ae2ca49429252097f509ed2d7ae710540d017933b79860215e48c621bef83",
  ],
  "executorch": [
    "sha256": "065372cc310626f510ef3a6649c0cd379df1eaca23b3f168f5d805b1d5febb87",
    "sha256" + debug: "62d2001024674ec8d8ad29cfc81a1dea39d299f341f37daacaa5d88a1944aa01",
  ],
  "kernels_custom": [
    "sha256": "59898d624e2359e8cd918d756e431a55184701cd36462fd8f2d7ed584cef20ef",
    "sha256" + debug: "4b8cf1a31a67f7991f18f097674824c9350619bb71d6408dc65743887c97df28",
  ],
  "kernels_optimized": [
    "sha256": "92a7b42ebb0a544225582e5943288b30f2bafe7e131b3b4eba36f56daff5ef2a",
    "sha256" + debug: "86b53a9729d1b03aa6f6ed9649eb6ab987389b82da1d80d29dd53f7c7d57fad7",
  ],
  "kernels_portable": [
    "sha256": "1c828d6c03de376ae59a2495a70e30416dbd81fe07c4a513dd20f07f322f811f",
    "sha256" + debug: "13a96311170fbbb9fafbbf68f0ed1a53269e6dccf67e5ff7be1088573fa694af",
  ],
  "kernels_quantized": [
    "sha256": "b00d04de07079df00901f2dfa6b37e04c2061ee82c4172103c70f5cbfc169d55",
    "sha256" + debug: "1abfe47ccb01c0148a7da6070e986518ba1bd1125bac5a145e53ba2de84ef8d9",
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
