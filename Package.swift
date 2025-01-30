// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250130"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "dd45ad84fca1147d3b225a9e12ec496488652b89e1019ce2c8e2d1c967384c10",
    "sha256" + debug: "b69e3202a42f74c09ba54eef416f63522d3a6603b7c4bd8c1878d0d810b360b0",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b67e503a37c48eedc8dfba739216e476bf435a20caa93a1061e7003b5cd3a29f",
    "sha256" + debug: "eeb356a8267b1e3c879adfe9ac0c66947f14d6582d70a1875df11682efe3d107",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "de79dc5e464bbd820e4ce635c9dc2d06aa1e7581d94b30a8cbaa50b0ccf66c6f",
    "sha256" + debug: "97b973b4aa1662a007a4e7bd0a9fb7a07556a9e3956f1fa69cba36389e33c8bb",
  ],
  "executorch": [
    "sha256": "a801452ae80d0abcd335473203306a897251557ef13c4e011e1a3d5e2815cbdf",
    "sha256" + debug: "740f3ba51e63a7d81701ffe87196dc61e71696d1c589348459faa93ec5475fa0",
  ],
  "kernels_custom": [
    "sha256": "fa5802f4a2ad0733499a454f20d2866821567cd7854d2533d1094082ba66b1c1",
    "sha256" + debug: "986d815343c839ffa6c86c9a5719d93534513da7daffc3af64dc6e8f5072cf16",
  ],
  "kernels_optimized": [
    "sha256": "6719ccceebe70d184af757b8e2d1cfbd8c8312ef211553b20fdbccc2457094e3",
    "sha256" + debug: "473b9c02681d28d19f8639457a716244db727e75fde7f6fd56d111cdecb162ff",
  ],
  "kernels_portable": [
    "sha256": "038183f5251cbf6b6a68c27bfbe2b49de90dd20928f1db512ca181285629d75e",
    "sha256" + debug: "41128531907c7648dcbf9bd0f76504e1c28a6a5bd7e2a9a065018365c2630a31",
  ],
  "kernels_quantized": [
    "sha256": "34272450f80531dd081b71b103eb4ac4e153e09443e3551476418424f9852aaa",
    "sha256" + debug: "c8ebaac48c75df9df38043e78032a2358bb2a0ca823e8dd342195074232f3f8f",
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
