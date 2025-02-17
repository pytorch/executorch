// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250217"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "01b085315fb5a79dc06376eae3bf4e37a64ac68ba74f48e9cd043e39269f40c5",
    "sha256" + debug: "a2d71d077083db9d43055c0cf0654cd7edaf2ae60a554ac862ec971fbd41236d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1024380aa3ca42a1845157fb3f48e3e70a1a81131fe1beebb363f938e80dd76c",
    "sha256" + debug: "4becb208d29366018d835984683474ec3069770f15bf6c3cd7c3361fdc8db785",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "194effadabdfdb3aba8621fca834e3c8c9da065161a6bd34789f37398de4e9d0",
    "sha256" + debug: "15b603237e0c4f4860f322f3d7026cfe3743b3c002624d5db96c22225c734269",
  ],
  "executorch": [
    "sha256": "41f1a0c5b09523b59c7682e438ca2b4b7533973d2463bbd8c698f070bc189988",
    "sha256" + debug: "7a12c44b8906c5b1013ae5254235d19b4fe501433f53a1062ea0cce1582add6a",
  ],
  "kernels_custom": [
    "sha256": "995c0512a58f404036965398d50622bf1da71e29ac5724bae7aaaa44096a31a3",
    "sha256" + debug: "ea0a546afaa1043320fa3402e30f14268b063b80d29707e47aabb9352d8d25ad",
  ],
  "kernels_optimized": [
    "sha256": "5981483f9d610894c0883199a80e0066d3f6586ca3b34d3caaccfdfdb837ff2d",
    "sha256" + debug: "1cf4af06ed8cfecfbffb43318b59e35dea5763788e29010509c160f1768db457",
  ],
  "kernels_portable": [
    "sha256": "ce2ebd29abe95cb3c3ceac4da16c5488783bf1ddf327ee2c77ea1d81c906d2fc",
    "sha256" + debug: "eb8c6f6d957b90c65bda99b29bb8aebf5d5bf125f902a3eb03e3721428b1c550",
  ],
  "kernels_quantized": [
    "sha256": "983fc2c505190286dbd5c6c29fe6c3fa0104be0d92c45b39ec340e9a396d81ae",
    "sha256" + debug: "a8086ae434ba3453e63c5bf0b52608f541c505ab951e39906c1931cf3fff8a68",
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
