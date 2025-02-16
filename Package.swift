// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250216"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "45036e6643ee83fe5bc375defa4dd98a913c75479121bd37f32613ca49428778",
    "sha256" + debug: "895bbe1180c0bed63dcc46b8dd80240e5500fe8c33d96276a6e6f50e4abcf1e7",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "def123ff03367fa893cc291c80b8f704f4061f9bda0a0a07f2d5dc9160e6a147",
    "sha256" + debug: "013b3f1b255f9753c71a2ef11a756a9474acefb00ede2ce76121d4db4812e292",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "805ae5c37e92f7d57195fda9c014d766daeff2b2fd3707637a04c106d1f3b85e",
    "sha256" + debug: "a9405687cf1ddc0b68af8b18e5158cf83572362f1e61b94f06a654a7ff8c44dd",
  ],
  "executorch": [
    "sha256": "7335b447b16a2d749550f49a220ac9f499fa5a21beb44ea80b06d040be70aca8",
    "sha256" + debug: "9701b625548d3310e61902bda078971c11b3ab5b2df0174cfc2ea4eabc48c074",
  ],
  "kernels_custom": [
    "sha256": "c688e1c17865186fb9aa12639dabd898ae302bda21254ea816260790b076927d",
    "sha256" + debug: "b1982d4313cd0df48e27b58e0c8524ad8ea63b7ed41dd682cae15ee9aed6e9a6",
  ],
  "kernels_optimized": [
    "sha256": "5e9e63f2378ad00cb84be20cc45da3e45b26c220ab737e53c6b1efcb9638119f",
    "sha256" + debug: "465980871df3666bf21f043d144017ee49caf5d06e7422424ebc040f0c416fd0",
  ],
  "kernels_portable": [
    "sha256": "f09c3630c94e8233ad9292d55d2f872bfefe00b621bf2892bc84d9981f738717",
    "sha256" + debug: "83f4ece437f30ea1f64c45ad47321ddf83c01125d0adf88d2824ede113a31a58",
  ],
  "kernels_quantized": [
    "sha256": "8ac8edad16da194baff36bd463887dbd291cb1126077e89928cc1413ce6e0751",
    "sha256" + debug: "97d975ae93a3cd94270eb6b10de15138b6a75667ff83b7104eb3a0824eb7603c",
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
