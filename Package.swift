// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250127"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "049fa808d711b6da91f2f016961ae63222a310218538a8ff3f9f885393c9299f",
    "sha256" + debug: "9f0f2c82bb08d4c9699ff7919f30db0a00529bfd1c1ed11552b5d402596c9684",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "cf9e1056c604329d7384bd31cd4825a43fa98472e0a2c63079f048971309e833",
    "sha256" + debug: "a51ac9616e4bb6f6478b80f73e255f982afb0f8f336d9c600a05e28c74893056",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "06e49a19f6ccb16752cd1472c3399a35d6377b43964aac5132da9d728dbf15ed",
    "sha256" + debug: "0d5a71bb8c44556b4d85984abf5c0acbd32baf6c692c99c040703f1d66e1f6bf",
  ],
  "executorch": [
    "sha256": "a8d173f624a7c9a85c96994dd0c3da953e5efc36b76ead9d510c8ac99833b90a",
    "sha256" + debug: "f4c36438b727b3c62ef02d84cfcaeafdd86d932fc2d49baf8b518df2b42428ae",
  ],
  "kernels_custom": [
    "sha256": "5302207839fb3c314fae20b7e288b9bc0d2cefce90d80701a5f1f7bea2da3124",
    "sha256" + debug: "0214411e72a9166d8fb454b21c0cfdaae239846d72aa887f08baa6dd12ab8c44",
  ],
  "kernels_optimized": [
    "sha256": "f84c32cfb041ef4306d26dcd41d00643325ec1a7cf58b3bc97d28d759ed9cf73",
    "sha256" + debug: "0447db4134cdea79dcf5b5f0c1257ed00d45aa012aed6b568e1f1c414beda5d9",
  ],
  "kernels_portable": [
    "sha256": "a5d719f42b32aa0f9bfb8803c425bf2415377e4d3972414b393d62c109c24710",
    "sha256" + debug: "95c3aea38a37ec9ef11aab2dad7d06c53700992e6a3f77f885e1cab75917777e",
  ],
  "kernels_quantized": [
    "sha256": "9ad71cd27767b239e50bd5adb8421fe48dcd5e41728d3f8f832799e286ce6569",
    "sha256" + debug: "3a258d5865653b4dd05f11cd0af53b23e8437154da6cc81ede2f2c3b2b792978",
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
