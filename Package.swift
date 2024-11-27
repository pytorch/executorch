// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241127"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "a004453d096aea20f2ccf95fdf92aec50625f590eabc57edc73ced2696d6eef5",
    "sha256" + debug: "fe7d7c9dee4b28e263b4fc0831c8d947ce68d50aa0b140ef227deb94bedfefcb",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2efaa27cf76ceaacfea427283200348a8ece443ad7d195199b5df1efe7a25b48",
    "sha256" + debug: "2e8ffd3fab661faff2d315021954fdd1d60cb1e9ba350412e6548aaeab4fc39d",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b9eac76921bb09d8efd544c71043279367a030fa9e737f6096420b5a0bf7fe1c",
    "sha256" + debug: "af0ad594f4ec6420d91aacc23de367db840250a6d2486011429b6ca3a752ca58",
  ],
  "executorch": [
    "sha256": "50c4bc4fde6893b8821f4828c8d7f2ba844d6447be861aed4e720defd47e348d",
    "sha256" + debug: "a5bc0dd7782c03147d72f8205120cdba935bb5142d2fd1963d8572dc82cf309c",
  ],
  "kernels_custom": [
    "sha256": "cbc6ab325426a588e0e8d40cc7d3781f33b98ef753216c3749619dc1706b4320",
    "sha256" + debug: "aa86ecdc35fb480fe43d2c6622958d5937af4e7ee38c2578e07a735d8c32cfe9",
  ],
  "kernels_optimized": [
    "sha256": "761f2f8b9f1e256213b0d1ce455128957e7a049e8892b0f325e263822367743f",
    "sha256" + debug: "9a8c0d1cb5e252a059e1373bb7abce836bb5ae6fefd9be54ba1a07084f359b5a",
  ],
  "kernels_portable": [
    "sha256": "efaa9eccea83e64f6d6102f51069161d5b09735954661c208768369fc7216f32",
    "sha256" + debug: "67c0d9dbe8e3b6dbfa6803dccc6b1aa9c949aef4236d40d73abf61ab5cadd9ff",
  ],
  "kernels_quantized": [
    "sha256": "5cfb846c8bec35ba0225e89d2a59df660190b47c9bb2638ce842525407266517",
    "sha256" + debug: "7886e2b5ad978a41f84e69a58aae73f7f90e537cf0dff7de050f1e7963c92b0f",
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
