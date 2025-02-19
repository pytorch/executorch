// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250219"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "381dc903c0b441d31c9d9c116d16387332c2bdd39487e935ed14a461cebca4fd",
    "sha256" + debug: "3bcded9910e617634b1068bc3f1f57b94b86e9df808eb1e4d5badaad6e451f1e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "ea7b31f92178f596044106d3f186c3913850ce087308da02b372169221ed7b24",
    "sha256" + debug: "ad816e6b1d5d4e968bb0c09eb801fba7d0782340df072b2927126351be240a66",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "60be2dda8fe7ea493fb8e4465558aceb4f0255f6f0ce8978f96fa0a7167c32d1",
    "sha256" + debug: "10906fe2c0bd9b7938b80d0f45d42f6d86ce4f39e3b7f77ac8333dc76f61a5fd",
  ],
  "executorch": [
    "sha256": "cc3c2085a533b81f8aee9e6d7eb1995572351ee24be218027371edb38267439b",
    "sha256" + debug: "c22e35dda4b98f76ff1832836dfb73a4cbbec1df7b17b3078d735c79d273cd37",
  ],
  "kernels_custom": [
    "sha256": "93e0bfb49ec1d5aabd8cd233054885653cdcc4a2e8b476914b8db97e00beb228",
    "sha256" + debug: "681a9879e93128c1f345a9bc64952ff454f98207be900830f4c569d1c06e9cbe",
  ],
  "kernels_optimized": [
    "sha256": "305680ca158a8d0fc62999c338a3603e399a7b27ae1b1fcd04321c019ef7b5a5",
    "sha256" + debug: "81288b4f576b91185242108d429e0be0e34dd8e8461da62139282ed5ffcf9c95",
  ],
  "kernels_portable": [
    "sha256": "c972e58cc3c8669d1e8c91e357697d4ab40a4639054010b6103cfec0ff7f3fb3",
    "sha256" + debug: "b5c4bd3a019670410c71674319bb8cba97af5c3ac4a61280ae1cc0867c961e49",
  ],
  "kernels_quantized": [
    "sha256": "184318850c4de9aee333515c58f307f1d9aba97ed822ca21ed18a3ada4b9e12c",
    "sha256" + debug: "5a0c45840c1fb34c94c1551f8cd8e757f9cbdcd11327fffb42920f8d88c9a8d1",
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
