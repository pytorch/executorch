// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "__VERSION__"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "__SHA256_backend_coreml__",
    "sha256" + debug: "__SHA256_backend_coreml_debug__",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "__SHA256_backend_mps__",
    "sha256" + debug: "__SHA256_backend_mps_debug__",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "__SHA256_backend_xnnpack__",
    "sha256" + debug: "__SHA256_backend_xnnpack_debug__",
  ],
  "executorch": [
    "sha256": "__SHA256_executorch__",
    "sha256" + debug: "__SHA256_executorch_debug__",
  ],
  "kernels_custom": [
    "sha256": "__SHA256_kernels_custom__",
    "sha256" + debug: "__SHA256_kernels_custom_debug__",
  ],
  "kernels_optimized": [
    "sha256": "__SHA256_kernels_optimized__",
    "sha256" + debug: "__SHA256_kernels_optimized_debug__",
  ],
  "kernels_portable": [
    "sha256": "__SHA256_kernels_portable__",
    "sha256" + debug: "__SHA256_kernels_portable_debug__",
  ],
  "kernels_quantized": [
    "sha256": "__SHA256_kernels_quantized__",
    "sha256" + debug: "__SHA256_kernels_quantized_debug__",
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
