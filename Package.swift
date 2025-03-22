// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250322"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "1c1a9269bf95e5e20d8321feeec89f60e42774b63edc18583ad50e651c25d3e1",
    "sha256" + debug: "b10cf9f01c53bab8e2929986809f56323688ce5134d13f54de310cfccb7883e0",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "83aeed7ab3344502ae6211dc6150b91262a6db778434fa19621a04e53d12b1f9",
    "sha256" + debug: "9b8cce4acaeb456d4d262b9aff90b640c1f45b843b890a9a63b2cdf615daa4eb",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e03bdabf65ff50635a300045e9ee2a242097285e2103bb0fca25de1acddb34cb",
    "sha256" + debug: "d78dfde2b85c06ca200a3523cc1376867fe21ca60ad0e85fcc8cf0b22318bfc9",
  ],
  "executorch": [
    "sha256": "cac28113594b6239de081b3148fd48db1257fe8d1f05aea509dba953043daeb2",
    "sha256" + debug: "3b16c2f21cf0f6d341b7be20c706656ef5f74c04f62482968cd9b78d5c88bcd5",
  ],
  "kernels_custom": [
    "sha256": "b3e2a742285cea202afd28c5a6b0d2979296572e82b91f0a5b19dc7e9d948148",
    "sha256" + debug: "762db33cf1f01694f298c0a4a04b03ffc606a906ccf148d7220f274ea28d386d",
  ],
  "kernels_optimized": [
    "sha256": "95d38d4d70418dd64e52926383fb4e5798ae4880909e99ae8564f0af8f69b91f",
    "sha256" + debug: "20b34c794192761246ba9f9257805297b240fe0765a2f790a1083b7be920a109",
  ],
  "kernels_portable": [
    "sha256": "557c074152ceaf2dbe15c61ed0683ff782da77d9f9d9c10cf75738cc7fb52a40",
    "sha256" + debug: "65b1f5b4ecfc5139610af596f256db86b021aa4f90cded1511fdd35729c60f99",
  ],
  "kernels_quantized": [
    "sha256": "821b1ebb918b473894460ead6c16c607c42e55adb3b99c82ee6a1b476a811712",
    "sha256" + debug: "47501c12fdf1569bebcd7ec4d5137a36eee27da2ac0f9c8c26b0d51aecfbb676",
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
