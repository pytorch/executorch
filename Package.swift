// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241212"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "2f5760499b3dfe11fd4d8eeb380fb424a8062fdbc13707c5d174b04d6b9140ac",
    "sha256" + debug: "fa67bbbde65c0c7d8ba2bf99b672b43c5d0a5af06ba4f01bee491c08626914dd",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7f612f1ae9bfcb709094e80eedad5cfe43f04f540f9bd9d1303c4bb7945628af",
    "sha256" + debug: "b44afcf0772f439723153a198eb4d9dc4cb6d8e0382f6638e5b659b81f406ba6",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "74257beead1f27633faff30eb8d056317599b214570b4f0ffbb61c891787daa0",
    "sha256" + debug: "0c0c7d8a7bc5681a9efa57f9531a5a378f86bf8efc2fc2c8447003095f7d60c6",
  ],
  "executorch": [
    "sha256": "afc44ab53fcf39eaf9fc9523041057b357e1324f86f5b6e463e1a754cb258000",
    "sha256" + debug: "d285b7c1d0790da9e03a28384fe167bd525bf8e183c38770e107680867b7e78f",
  ],
  "kernels_custom": [
    "sha256": "e87615d30b7eed789925c18577eaf35580d3db89aa867d0dd0e8c01d97c25f8b",
    "sha256" + debug: "96868b449b000c6a904ac1784d7279269ec6b7975121be71c045433b62955c4c",
  ],
  "kernels_optimized": [
    "sha256": "a4dcd0a48f2ef0ecd9cdff3ea3354b2d37125189bbc84ee38467bcd0090b83fb",
    "sha256" + debug: "4710aa20b56f0ec67d51419fed0a0c14b3d562822be8096df6988504fa48156e",
  ],
  "kernels_portable": [
    "sha256": "cbe187174b780f6672924a11f3ce2aa56ab2639a904185d162acdb927e7b600b",
    "sha256" + debug: "12e67a02a046d060ca07d436e316018c4d7b7a79f7e8edc25412597cc4ce55d2",
  ],
  "kernels_quantized": [
    "sha256": "88f57bf9690e57ceed8cc3243d62b8784129e453e77629410195c525505a7fc3",
    "sha256" + debug: "bd7272d7374934098227787e4d672763f834f94d3e65f6c60e03fb6f41b85dc2",
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
