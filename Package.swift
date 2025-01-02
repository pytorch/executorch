// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250102"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "f5cb7410f1bae59c5cb208eb7633740ea4aff74383cd8e8a127a767c5ee8746e",
    "sha256" + debug: "65b5b10fa60cec1ce83623dd82d07c0bb78f074dbb4d8460e6db4907e0359111",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7882944ad101e149c702e520d96c77d2e6a5b4486d5e7b61c17c18b4390ad98a",
    "sha256" + debug: "59d4ac448b1477122ca0f57beec868de24b10d2c4c6b10192f97fa864621ecea",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "68dc74edc0ae88260d05fa248663dbd30bfc0f56aefcae2f29c2fc1150bdaa3e",
    "sha256" + debug: "381bf1a6b02a219e2dfe8859938a8c411071a6bce83823f9ca409abc843fd488",
  ],
  "executorch": [
    "sha256": "bf99a088556594b1afb94d2ae8e252d19ab272e2c80a5c0ac6d7362188afdff4",
    "sha256" + debug: "c3deac63d0106c4c1c5cfb480df2d3a7dc73384c643d6388c11d823d2a3170f0",
  ],
  "kernels_custom": [
    "sha256": "e15c3350852f2f40c5eb762bd8ec0b16b8fe26915a5761ef6a15faeb7e7469ea",
    "sha256" + debug: "b21d594bbae2202595ca93459fa40124919928bd7f5f232e54ba6602556ddb61",
  ],
  "kernels_optimized": [
    "sha256": "90d1bdf629bb652be55f484b6bd1cd92e300d14d45617210e5db6276e2e794ce",
    "sha256" + debug: "3340e0d7b0f99c17d3f7088f0ff8ef08d9f99f75de30c913c1e890865f39303e",
  ],
  "kernels_portable": [
    "sha256": "92e76d1a3a59d7d56175d5d72eb0515034ef93c2f341f2352b6c99c61c01fead",
    "sha256" + debug: "669a66a52ba38a892443e642e9765714a510833a36729098207267b1d2c32ddc",
  ],
  "kernels_quantized": [
    "sha256": "06b835886180d69089df6e0bfd4bd96ccfad8bb73f174edf39bd4f13cd29f453",
    "sha256" + debug: "cbf33c0a8dc67322366cba20dcc860f51f5b17ce8cd367d8f444873d50f03b21",
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
