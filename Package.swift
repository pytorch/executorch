// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250226"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "5b3461e81d8cd63925cc321e9e28f0fb4c8b903cd5bae0e5b07e2d7f70236894",
    "sha256" + debug: "b9b639ebfdbb62d600ca98eec270911f71ef839ad2325a8867094f9a3a4f8074",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "299d4ee927e1ed35680d310d4dc955c8e583de7fb3f28383ff794ad5fff7de31",
    "sha256" + debug: "bc24c0293395edec2810007e233373238d2e146ed4aeea6110cf7e88d2fcb551",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b1741e1a7f410ccfc9f6855a8278702e4472891efb684f8977da7ccdf58260f9",
    "sha256" + debug: "8985bdec7a19b7f54f33b6d2b351933ededb76917051c7a1390846fd80ccba83",
  ],
  "executorch": [
    "sha256": "581900138fe6130c51179a0c42d8f0acc7b25a5f8543e24358479cdff5208d33",
    "sha256" + debug: "f56fb7ad10f3d74d00690e9d29c9e5b101fed1bc53389f136c0c003f254a0bd6",
  ],
  "kernels_custom": [
    "sha256": "c470373f40b7b965c3ec717f1d8285236485c286810c0b2f0a38112247dde256",
    "sha256" + debug: "ce00dda57e66d44ab0edd1c31ef68d084793f3221474c22412ab921acf956507",
  ],
  "kernels_optimized": [
    "sha256": "bf3ed1729922fe38ce8f006da90791972d5bd6ce70ff78a0d75b1dcd51104e1c",
    "sha256" + debug: "af0d6352dd7950de73cd8f35b5b622cd7859d8edf9a9d6ab178792b9b150b17d",
  ],
  "kernels_portable": [
    "sha256": "1277b08d4c881e7a57b27c49016a29b503a18b05530f99f652515cbc5e1cc4ab",
    "sha256" + debug: "ad5082931dc988855d9832946c7ec8005163c5794c615f97da900961a2ced3fc",
  ],
  "kernels_quantized": [
    "sha256": "41380e168d8f05ffe49a000609cddf0a0810991bf3ef26e7916e047041d3353f",
    "sha256" + debug: "ae634eacfc801adae4fb789595ab71eadc258f4e00408eeab4782a5bbb1e48a1",
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
