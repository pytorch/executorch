// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250423"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "e466903c11ebdfcb306a0805296af6c13e7b881ed13bccb48263f1d16d2edb2b",
    "sha256" + debug: "3f557867e28b6befc433a12fe1b0a6fc45d814270598d16c8c3bb1d3d64b1eb8",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "ca095d1418ad99ca300707a1dc0b27a055f8a59f29cf8de9d881baf518fed3a8",
    "sha256" + debug: "61edf9ebfa44475a561baf8a6c6118846ee4c62eb55de57fee194660605bb8c5",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d096eac99358467de4016ec17588f574dce1dc0170919fc625b69e91a28136f4",
    "sha256" + debug: "e640e80b26b8261458f091da3dd6ca5fcd3cc420325c2cab91d8570397f238f5",
  ],
  "executorch": [
    "sha256": "34126648c7967d2cae2d2350ad17e0ebe2db6eb14272dc31694d30c3e7db83d7",
    "sha256" + debug: "516289d5b871067198688b31f7b476f837a736b19c06966f0573a64601a42aff",
  ],
  "kernels_custom": [
    "sha256": "d64c191db070f8acb0359d193028d0274a3e9b2096af12d7f59fe40d1ef96540",
    "sha256" + debug: "2a12475c5a460a840d695dde5981cdc5b8af07c61f5ca2d6cabfc222b8976592",
  ],
  "kernels_optimized": [
    "sha256": "deea4225daf9dbeaea37b5a67f19013e257474643c9a365511a1119a0f47b297",
    "sha256" + debug: "536d1bbde2e636d331b8501433d1ca08cf2751f50bde6cd61d3dd774d329dee1",
  ],
  "kernels_portable": [
    "sha256": "7dfc3757884bd7625789885f63e10e2748ccb2f162ff7c05c55426a74e9fb5a0",
    "sha256" + debug: "1625a4d08c936f3fab63993638faf145ea78c580eab862754200949a3ef77503",
  ],
  "kernels_quantized": [
    "sha256": "d4aa8714316fbd405867259c783a3e1bd4b163e15e7d15779d8b9977859110c0",
    "sha256" + debug: "9c26fb037ee75b0efe399e70807446e017fa650626baf51e2fd2a7be6c6b4173",
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
        linkerSettings:
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
