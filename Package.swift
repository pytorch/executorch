// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250125"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "01f5f5aba2f80064d5c0d98185e56f1231da4d8e68ad6960f864290d792efe64",
    "sha256" + debug: "588956cc578529451dfab75d9df2ee74f7707686b4b0f1df396c8de522c8a68c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "4cfc8a225b7cead07b8accbb398449dbce7b2d7cbfc0eb3aad3cb017ab8d9508",
    "sha256" + debug: "c431d1f2dcb37b04f55b72a48b17d2efbf0e62f579f92afeef663da28b94054a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "88a4497c880da1bb44155ca261e9a5c9378e36b5e04b03b34b7065a16218d3f6",
    "sha256" + debug: "50325caf1fec96d2ae519de79b67f9a1fd9fba88650678dff414dea15047b182",
  ],
  "executorch": [
    "sha256": "1a1037834e2019961baad993cfe632d45ac0a4a383c93759c364c35d4a575624",
    "sha256" + debug: "98ba4c60a779856af0358460b57e6d5ec2135ef8cfb6a42472ff7b19f0268557",
  ],
  "kernels_custom": [
    "sha256": "24a58ab5f628a9f3a35ea16935844f2ac153c882b5cb9eede41d02c2b9b8384b",
    "sha256" + debug: "562de4b558bdd8c010c829bb15d8cc991f2c94e1b3fe949df2db78de12f6d616",
  ],
  "kernels_optimized": [
    "sha256": "0e4406b45d8c064b5e943a43c0e79620b47361b03d720d71f890d835e2d9ecb4",
    "sha256" + debug: "4ee7b05bbcc8c7a0c9d9a6527a10caa8f813cbfedb07967014b1c708e7e57253",
  ],
  "kernels_portable": [
    "sha256": "1d48645133563776dbe0fbc40a5ba5cc0afd1b28309563b1e74f4b3293281bc9",
    "sha256" + debug: "168834e1d97af1486c549ef7c2f81b991e4106f5b0e9a139eecfc5486e4a0a18",
  ],
  "kernels_quantized": [
    "sha256": "64916f79e8b056af292a8dcc48ccd060570e4e635b8f97042a1f30e77ce8d449",
    "sha256" + debug: "9d1f93b2e6dcd84037dbde30964939cd6a08c44e81e8155381fae4a7a88c026b",
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
