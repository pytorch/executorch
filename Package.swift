// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250316"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "8535812e3a351699467ec98d8d17ee841115abe43e2e5cf555bf8ebdac67e353",
    "sha256" + debug: "54c82d19daf840549e05aa1fc994d2bc32babec0dee71b04e0230b27d7f9285c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "04b25a1dd1ba12e8232cb0adba34ba50fdc71267575c02c65962096e61381fe7",
    "sha256" + debug: "6760bb4f52b775360b94c6d8dcefe48c39bdfdb86761aeb5759b3e9b01f5af6e",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b815b32dd2d6565feae8bbb3cebbeb58ded646c3e8038844117bad2ab86fe297",
    "sha256" + debug: "356956bd33cfba24aef56b383f44c82f146d0fffd87a32f7a8900731bb5f0ec1",
  ],
  "executorch": [
    "sha256": "1dcef85c0f88190a03599a2e44883c106562bf557c92afb1abcd3db5a1977c06",
    "sha256" + debug: "915768afabd78eb176b406a8b953d50ae32506d0973789a8fadfe3f3f76f1cc2",
  ],
  "kernels_custom": [
    "sha256": "7f94ca447190b4152e310f370b5a80898f975601e7187226e191d121a762d2fd",
    "sha256" + debug: "e9a896ceb743488f75801c17d65a715019e69702994199ad8f82c71c7043e683",
  ],
  "kernels_optimized": [
    "sha256": "1ef2bb38b9e95642a4c99c69f9dda3ef013dc32f4c7d2c61ffa0ee76f2cf3020",
    "sha256" + debug: "331792b654f5012c211b6e9dcd8cbf5930dfb3872637a0a87a3bee4f4cd52227",
  ],
  "kernels_portable": [
    "sha256": "42b0540571c831344dd8a130aca5a50d0f8f726e23b132c74b6725f6a327cd5f",
    "sha256" + debug: "352502c0a088c58745813057b0bd1f5b6e3bf37e72914b43e0a6c40f9a2f058a",
  ],
  "kernels_quantized": [
    "sha256": "d094dfa83e8b5c786b8d3f3f5584e4a42e5dc74ac2303bac8c7c62c0624282b2",
    "sha256" + debug: "5a36d31ff55c68bc3ddba0f4f1e8cc173d95618a998b5713e3c9cc18f0411db0",
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
