// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250513"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "b0bb573f5ac9b7325a6e5d4efd413b6e5fee5a1abff7599ba27bd3958b6b8228",
    "sha256" + debug: "c4996af5672e4267dbdbd56b56b22a4a5d10e48fbbf8aa9a8b787dbb8353029f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "986e5b9dd754ae75b52d25024b9dfeb159b80439c1b58a1ab252f6d3ec1ea333",
    "sha256" + debug: "2254d5b56d22896ad30c878cecdd7d443fd10b9a2f77d32d3d722d4829b92aa5",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7ee4326c77d24822f6b992fcf61da2faa059380289e910dd63a1a69694ea77a9",
    "sha256" + debug: "f8a44655cc6d42c199ac358d7a3b31ab98bad67b4eab5035375a3257a039c665",
  ],
  "executorch": [
    "sha256": "4cec5337a78211278eeddd68deef987fd6c881b12fe3c755ceec6a95383df4ff",
    "sha256" + debug: "782f2efec4e36aa3201a93b0bc6d1b773ace4db627805b232e84d75a5efd910b",
  ],
  "kernels_custom": [
    "sha256": "2a0d0ed169b2eb01a5b7d0bdaad9e7e45bb963170742996471a379c51ec92cf7",
    "sha256" + debug: "5bd5033fd01eca4c9e3c99082f3eecf28e529c7744677cd0a6fc2aef6cfd79ca",
  ],
  "kernels_optimized": [
    "sha256": "e85312ac5889da79abc7a4397fcd15512b325bd1270de4b6ffb3b7f3ddbd99cd",
    "sha256" + debug: "3689cb778ce4afd5877081327b1c887905466098246e3549ea054d28cb03b9d6",
  ],
  "kernels_portable": [
    "sha256": "8b657c4fab7851a4c82f5301d0df514b3a632083b29351d75d85973fa004ccfa",
    "sha256" + debug: "e53dd9787e95938ea1e11612fe7e6911a57bff9a679646d9d72b0ae39778e2df",
  ],
  "kernels_quantized": [
    "sha256": "7302d4cd2cb0582c6a3dab315006d4b4763a7cdfb745eedff5223dea2bf2ea3c",
    "sha256" + debug: "478c6a3d3f0f968f2aa7114413a782484055b2525098a06ce465e029d850e046",
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
        linkerSettings: [
          .linkedLibrary("c++")
        ] +
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
