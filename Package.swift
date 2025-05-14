// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250514"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "629fc36549d94bc5076b4f4bd3429da4147902b3a43ed94319273c16c1f7400d",
    "sha256" + debug: "9faac0c506bc7daa877a67ab53d9a53352d364060f148611d3841a2721d13be3",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "511699493880a4651f33b108cfeba743070b4d08034c5ded9c175c9ca736d18b",
    "sha256" + debug: "0ff8849698597435abe2eb48347631727241ca7d41498978d9fb9d4d88f3037a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "231cab63cd6892d00802fb5735a47c8175432bd1f7cf05688bcfa4e32e114806",
    "sha256" + debug: "e8ea391be31c3899e8169f21f34a4b15154fece77a64a7e3f0720b8257767fc8",
  ],
  "executorch": [
    "sha256": "c3cb16295005b6d8241f2b7528bfcacb70b139242a2d4624cc996fb113808b5f",
    "sha256" + debug: "cb6e3a1668863f13d4f116ee7c7a9e87c0bb7597c598a885ff07803a80432c08",
  ],
  "kernels_custom": [
    "sha256": "3603c55473acd1b14d836c37b1c2751015a0fe3100fd6ac1e97a7f5f40143872",
    "sha256" + debug: "89d24beba23c458fd1f60e6d45d0cced43ec154a36a561c20b1bc333adb0bbe7",
  ],
  "kernels_optimized": [
    "sha256": "79ca219df69c657174cdbc91b65b18f389880a0e5869c22b8bf0dcceff5642aa",
    "sha256" + debug: "6607cfc513bb48324d110e7e96acdb73ee51548279f39dda962dd2126403569f",
  ],
  "kernels_portable": [
    "sha256": "82643b6f700513691703784f888ccfe189a052b075516681cb060e3937012df7",
    "sha256" + debug: "ed8b989ea5345ba75638875c8f323ddbe4a4a274d87a26116ffb640187f080f4",
  ],
  "kernels_quantized": [
    "sha256": "fee3b10eb7db0392de3caac558ce5baa5029dbe2bbaf3056872a59db72faba9e",
    "sha256" + debug: "5d13138e1403a38191d11bbce4a776d2627c2b64819bd19c7e2dbae5b85b4036",
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
