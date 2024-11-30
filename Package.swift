// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241130"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "27adb56c775bac1ebc53768a1da37347bd8c8bdb318ee9f35f28204358ff09f8",
    "sha256" + debug: "0e270bfe066e059f98d0965022217a1df7ecdda6e634fcb2c33bdb180f4a1f65",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1cc9ab83f1e27986f02e9f07dc8e08faf0b270b319cf3d0f434a60cee4f029bb",
    "sha256" + debug: "29bd5c814024b870fb7161ad9cee44c96aa4b2554499948f768fc4efb3856b68",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b5f1fdf94c8ede1ea3b4a664eaa49c8926abd8d6835c5a8e7f26c84aac6e1de6",
    "sha256" + debug: "aa4cd9cea070401e641d22c7bac53564101cc537fd6caaeaf15d72cfd83d5952",
  ],
  "executorch": [
    "sha256": "ad757ce1bb174786fb57be14fbe6ecdfbfffcce99fe0541d6924bfc1081a846d",
    "sha256" + debug: "f2a560c9316c2298198e0189b91388019b99f8b2e284644b5690bbe5c268436c",
  ],
  "kernels_custom": [
    "sha256": "f524581ea4f0dc4128c44574ae8d1955f42c3b91ca8debb39e47299dd0fcf74a",
    "sha256" + debug: "957094a923420657340f1f71915c3590e1d7de53b2ce0c94f6312b302883714d",
  ],
  "kernels_optimized": [
    "sha256": "ee6454420a5d41e5a5f4086e3b3ef3d9ca9cd39032c8e560b27bf40baf5ca70e",
    "sha256" + debug: "aa7ebdd9dc40425c4bbb1b0214a730d80bef30d01d9bd6ef365e60312fab807e",
  ],
  "kernels_portable": [
    "sha256": "41d01d59e16efd21c3dbac65f6afcdfb80d1ac3a4aeb31f8791add89dd829557",
    "sha256" + debug: "446be0faa0f8e2eaf2dfb5720dc4f607973fba411269d57e1ec80330d673c721",
  ],
  "kernels_quantized": [
    "sha256": "fcac167850c9d87730107a2da0186b30e6290eec90483f3c834a7b5d6e253e7d",
    "sha256" + debug: "7fd9c6a5e69118d008a4f51464506208af6fde8bd40da3b88225fcfe30a44df4",
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
