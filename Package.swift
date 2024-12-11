// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241211"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "97f61e0fed2f5a6f5db3d645e570e923540b9bb75ecf4a18db42d29827b94226",
    "sha256" + debug: "c68531571bf6d430bd982fdbaa546738d6acde1f302d2d9ce64d13115b6f83cb",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d7890a8a95dfc9fa4cc509920c697dd27dbb1a598d20c7f9010169b2d8a202ae",
    "sha256" + debug: "e8246a26d28b431fb2f7df441d99b70fa1154956a1b2682a538a1a6dd96c781a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f8b40b40aa41b2e5764e923c9c4ffab637acaea6df6aecc92728ebf7c38f99c2",
    "sha256" + debug: "74bbe9cef73f47d1ae57d4987d9d5c9a08af991c26cfaf8f055b363a5475947e",
  ],
  "executorch": [
    "sha256": "bdaac4dd824f61160872f5acb08b59fdee26b3c7f2fbef3103fe933b16a6989d",
    "sha256" + debug: "84e54924970bcd21518a2ab91fe131135731d28b4cfa3859cd2b4831a75138b6",
  ],
  "kernels_custom": [
    "sha256": "45fbe1ef87feb237eab51228a7f31d3ffbe2ba7fa0a5e365a7c26dbb8b00e1a5",
    "sha256" + debug: "d18a6eae68d53dc7cfeb6a54c7c569cf37821b19c5a292fc33fceb6b5e7f5481",
  ],
  "kernels_optimized": [
    "sha256": "bd2551c3d0f9f8905e811cc5f3a00556f39692bf8e11e9c51f1c468a3fba2ae3",
    "sha256" + debug: "8c7cf4dee7fe35abb3f3b2abb4c5afebf0214101ea5fa3ddadf7eea304c251ba",
  ],
  "kernels_portable": [
    "sha256": "d91a954d9c2eacc442630b8cb58649136bec6538ee643899e45d8ca841098a65",
    "sha256" + debug: "b8a4e17f0459d64db5bd13161acd8735ea2158d4abdf889fc9f6f8895387f132",
  ],
  "kernels_quantized": [
    "sha256": "031677fd277bd02fa24ee12b22f8b4ac965a154871a6a1f0df0090e8b02901f1",
    "sha256" + debug: "1f69d5917db44f9e51eddef67f22f4da6e60868753bfdadd762cb76fe510ba5a",
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
