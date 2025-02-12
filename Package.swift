// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250212"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "41dde80d40f7446afc6c4cf41bf761dd8b75cb6e0c358acdd0f5aa96e6fb8f12",
    "sha256" + debug: "3c08d75cae8e957f1ab2a56101b76ef4b31a97515ab76d96c5ec3c34d252d4b2",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2688fb2798835ab292a505ac197c0f0bdd443a225d17172f0ccdd820baa7e873",
    "sha256" + debug: "73fd53f3f3bdb600f6fa7a9364e4aa289670e20f185348d1f22be8fa4c82e8e1",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "aa22519c45e8ce5cf46813214ecaef317371379aa614b6dc3aadd932baa41af4",
    "sha256" + debug: "2c4bf4e28bb2c447c314b6a0c5bd0674b59d9945140054bb060b5967e6dc3ff3",
  ],
  "executorch": [
    "sha256": "7b8229e275bfc06946447d192d7a3cbe3ef9237878dc85c2dbb947fdeefa8332",
    "sha256" + debug: "390a2f689bc557aa5cb7a58cec3b9a5c776b06f317083c289a29b9f245f3e1c8",
  ],
  "kernels_custom": [
    "sha256": "a9fd4d31566e77431eacfdbb303d814f9f0037ac7829559f6fa624d3cbef48fc",
    "sha256" + debug: "4c0e61f9def90bc948803bf1c8bc6217589bc998619be690cd706a4a7578488e",
  ],
  "kernels_optimized": [
    "sha256": "3ef63f258b088321667ca3a1b491f8c86495372f5789c766478c90103f36e385",
    "sha256" + debug: "e35be7f58773505588516033a3966ec1e33a7255be4033c4d317309957753595",
  ],
  "kernels_portable": [
    "sha256": "d09cf21a51458547fdf7ecfe51a491f8eda239957b3d41e831ec0c662b7aec67",
    "sha256" + debug: "cf2f6ff882139e41c8e73e745c6a74a316130d5b0f029d3011d6b3e98b70d7ee",
  ],
  "kernels_quantized": [
    "sha256": "b4f362bd4f39a3104ce65b401f4fa72eaf44ecb6c21ff23da223169206c0dd95",
    "sha256" + debug: "be4ec136def9e7024b6afe3d9bce1608eccad09c8bb55f6ff97348b299d4d490",
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
