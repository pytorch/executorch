// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241222"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "267cca40714fb84090579f0f2f009902b51d130c4376d3ad6aa58915e743bcd4",
    "sha256" + debug: "8b6e2d89ae4746a257ef291937cca94d3debe58363d8d12130fdac7670c7ec0d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8bcc8a475331c4e0463224fc4f53b063e4736416a3fa9d16a7a0634bf09fdf40",
    "sha256" + debug: "d5b9f2c7f431e4a30d7436d51400ed9f6d4bc53d10ff281817bfdcc266a8663c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8bceed609bd5b86d571e547ecb5a5704290045388400e671f4d0b23487af16fa",
    "sha256" + debug: "4b9898146f91432c9edcea2f9f93610a92c21159a4d2a6b5c93f2bafb56603be",
  ],
  "executorch": [
    "sha256": "eaf2af03d35b6f6dc942377d83c48af86c1f0ceea86c7a66b8e62730ca62e61b",
    "sha256" + debug: "73fcf201aa4e77709f70b2e476187169dd02a6af9ddc998a0fd23a57b3ee36ca",
  ],
  "kernels_custom": [
    "sha256": "4a37ea11af85d216a80c5dba0810e8a863b553b43620c5d7de1e45c3d942c10f",
    "sha256" + debug: "f31fe07ef02802283c716cabd9047c0b9f8eb262648daf29587000effeda6f7e",
  ],
  "kernels_optimized": [
    "sha256": "c77731f17d3c5c0d18394be61652670263a3e66758c3638921642af6b7c8bad3",
    "sha256" + debug: "98c019d8870b013e2f73407d7c0a55b872532aed8c76add6c28b5741cbbd72f9",
  ],
  "kernels_portable": [
    "sha256": "aa190a5c9dd128e649d63cf854be6d693d861c89e916cff22111225741439fe0",
    "sha256" + debug: "20db40c8ac61fa6c03d900ec14db78a2c90da9f0549516d0371d801773577957",
  ],
  "kernels_quantized": [
    "sha256": "6c55e41503ead5b34bb2acf2857d7c1a9b0532f2bdad499c915c02fff87e9c5f",
    "sha256" + debug: "66c3e41c7c44e8645ea876ff70e3d1370f484fabdb43d0f32ee05c3c8ab20884",
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
