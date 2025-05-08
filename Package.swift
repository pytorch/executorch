// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250508"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "70782caa348ca5f7da89c26e371211d25e13740b66b30eee877ae4557b401f34",
    "sha256" + debug: "977bc0aad9760a5a97182f141ba2dfaae648627d0591e59c00da9f7cd244d2fc",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7c217303e2dcab2eb4cb5372f0dc8b2fdac58a435e9dc0c250dd7dfa22601208",
    "sha256" + debug: "d43a60cac34fbbad11d21043922ea464e4a5e2953069d3d093f47e56b3f40e84",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "658a43712008af2fac98e8c5c3dd3f10c0a61d6c39eca259e0d40c238af7bcb0",
    "sha256" + debug: "a5360a333305af8ad722531efcee09a30d3052238572dd4b429e4fbdb64133c0",
  ],
  "executorch": [
    "sha256": "c29b785e1e98be188a7c445c0a2f5ddbd449b3cc12b016deb34dc8a40f9e6144",
    "sha256" + debug: "d039d5267ba990f423213734b684b9c11f5f58b2547df8c66fbf1031bdbb3a33",
  ],
  "kernels_custom": [
    "sha256": "832192cdc6e2a635634ea8bb15e6bf73a55bfdc1d63e9f0344ab0eb2a636d1f8",
    "sha256" + debug: "7bfa9f13d05b1fc68858e496b8a0a6b4683c498021a227cdd91fc68df2479ccb",
  ],
  "kernels_optimized": [
    "sha256": "8a17ccc08457b65d6fd7d0eadb52480763c996588298b8d89edd1af21b914b9a",
    "sha256" + debug: "f868f2c7086484863eee573cfbe4b521b1ec8bebbbf0f5e37cf511904f66a1d4",
  ],
  "kernels_portable": [
    "sha256": "fdd5a917cbe43dfa63a2a8a5c00f985bfa6995bd2a9cb01a603445822d02b083",
    "sha256" + debug: "35c1389cfd3d94662c584f93e02b2e71d55bcfd6b5254d18727fb89c6670a434",
  ],
  "kernels_quantized": [
    "sha256": "82aee9eca87d3a3e36f7d27040eb1cd934c264c57e70e391945a059144f4c1fd",
    "sha256" + debug: "1cee9baac8a77bb9d9cd39ac96647173a28f89a957ea9c23c11eba6244912fb1",
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
