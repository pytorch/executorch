// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.1.0"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let deliverables = [
  "coreml_backend": [
    "sha256": "78d853d87be478696e56e658aa4ff17d47ae185a9a6a36316c821fa8b2d3aacd",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "custom_backend": [
    "sha256": "f059f6716298403dff89a952a70e323c54911be140d05f2467bd5cc61aaefae3",
  ],
  "executorch": [
    "sha256": "ba9a0c2b061afaedbc3c5454040a598b1371170bd9d9a30b7163c20e23339841",
  ],
  "mps_backend": [
    "sha256": "39542a8671cca1aa627102aa47785d0f6e2dfe9a40e2c22288a755057b00fbfa",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "optimized_backend": [
    "sha256": "1d84fa16197bb6f0dec01aaa29d2a140c0e14d8e5e92630a7b4dd6f48012506d",
  ],
  "portable_backend": [
    "sha256": "4993904f89ecb4476677ff3c072ed1a314a608170f10d364cfd23947851ccbf3",
  ],
  "quantized_backend": [
    "sha256": "8d35ee0e7ca77c19782eaea07a1888f576cda679f8a4a5edb03d80ebe858047e",
  ],
  "xnnpack_backend": [
    "sha256": "380e5185c4c48ede7cc0d0f0657ffb26df83cd9f55813d78593aea8a93942caf",
  ],
]

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v15),
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
