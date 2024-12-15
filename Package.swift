// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241215"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "f72a1b54abffda717cc3fb8e98a2eebd829d4d68ea9890d4ae86e155080b6376",
    "sha256" + debug: "b6fe8a48760db1b8c5c4102b882e8a977958522260738fb5bc25b1ccaae967e8",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "0866c1d1bba3ae1a3ca7d4b658065b86299f85529da90995df4c20ca63b8f0a2",
    "sha256" + debug: "611267a69bd50d55f1b532760b506a7246533a4fdcd4bbce28a418757c142009",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "63905ad676575a59b235634d62d1b9e468d1eb5adfae501e9a9dcc593b01827b",
    "sha256" + debug: "a2aa0ca619b3b52208859e0e714cac32ab08509810fd58d4ea07bf1d1dd1cabc",
  ],
  "executorch": [
    "sha256": "00138b82e3ee98cc46be9cd2c87ae68a34a35640767ca9692464a073d3443b17",
    "sha256" + debug: "4754b5fa0023783e5a6b5de7e817c3c0c96e9af5f2d1b7b7b7fdde91c87ed333",
  ],
  "kernels_custom": [
    "sha256": "945a578d2476908659ce1cf1fa2e33b29cf614884ab3a90e3d23f4af1eda88b6",
    "sha256" + debug: "7b452f0a792e3742a797be142d495a1567d7d47b5296104f8b3f7964a15d5c9a",
  ],
  "kernels_optimized": [
    "sha256": "9fc85f1c8e8c889582d2dcc6b1dbdeccb71c2ee9b213271f22f57fdc0f0a33f5",
    "sha256" + debug: "abe1111186d8304c31ac2c29b4e350419a190f77604d3e38ab3c4b5b9b71d3de",
  ],
  "kernels_portable": [
    "sha256": "504108b07c476828d72affb7a6f9444042314b472ad5dd695ae0ae919468341e",
    "sha256" + debug: "fd1888fecfdc1c6914dc93303a67d0e4a25dcd9b519149e2e21c1698849d06ff",
  ],
  "kernels_quantized": [
    "sha256": "3c60f34e4c73ea16bf110b8eecffe7ec2e6817dc84c3b20b1e810818b0ba49df",
    "sha256" + debug: "755ae660430eb9ffd31d12015b5ef4a3fb939f779d5285581e9ac1e41b86a592",
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
