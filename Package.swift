// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241216"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "5cc3e4c90829437f069e871bde4ee45e79723a5d9f08f2a4389425f7e4657549",
    "sha256" + debug: "ae5a9506091c6ede3306c64acc0c0477cac01d7bc633499b65e2562f0b55f18b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "050d9789e2a0ef542349bcfa1eeada50d70c77167c4ac5b6f4f448f00f17aeaa",
    "sha256" + debug: "92296dde5d3a032a374326c5b3673a8adde8d0bb61c80bc047f2f03d12c86e76",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "41d3e71a583c2ee4df9fc4b20208faa307ee9b5ba2feee3a1b0a24383612cd30",
    "sha256" + debug: "36165af86f137e128f542fcf75e9834438569efd653340420569b12d19b24bbb",
  ],
  "executorch": [
    "sha256": "6433f6c0b306d731a7b870b42d52fc70651adfce2da76215ca58cb185fbfb261",
    "sha256" + debug: "d58098d3c61b8a1b410c3c46a9f9cf85853ffda6d0a58a3af9c9dbee88b4290e",
  ],
  "kernels_custom": [
    "sha256": "32c82ded5de80602c1300899e9a5c61704573f3ee02d7a05a84ae7c3408ee1be",
    "sha256" + debug: "6f08e6406b5012174ede1a3aa075ca0de9e85c9546e74eb96ef823f1e2319286",
  ],
  "kernels_optimized": [
    "sha256": "44151516d09b61ea0daa0d97b1460792698aba13f974ddfc59595cd25e0af5d4",
    "sha256" + debug: "f37666fcca3d0138a88390f811bd383480bf93988e0e6670d3660291ae6fe05b",
  ],
  "kernels_portable": [
    "sha256": "3ee603441c1164aceffbce31844c3f77837ef60d835f78c5ba81501895c851ff",
    "sha256" + debug: "487853cf0a13788df793f7835aff107a4be9aa5565839fd8abf9754d5d5ddbe9",
  ],
  "kernels_quantized": [
    "sha256": "e186a7ce34c82416dcb6e0d0985594a3e5317e1abd717023fbabd2b5a0ece710",
    "sha256" + debug: "d13266b3410fc7424ad57d68d35483a97a8a708077e0ca3757572f35010e75eb",
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
