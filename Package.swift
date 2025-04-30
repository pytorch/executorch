// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250430"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "b6691e07581fdcdf396188f9cbf4953422543b27e99db30a9541790a90d63860",
    "sha256" + debug: "2562417048c9f05daf87ecb98a6ee732418af5d5bf179b3594b49509dc9a1253",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1e8e90dd57d55a400e6e50acab972b23e9b74838f5b6212120bde4b5630a39c9",
    "sha256" + debug: "ff4a0db9bbffd130208c4fdf1bc01cf1fd19bcccb6981e9bc3ea56b22a2e3d9b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "dddd9a01290d380df08fc3f31442ec4b59eb8d92f649d9b283ca202d1241fccf",
    "sha256" + debug: "86058020b471005713db035d73fb4e90be65706bc429842c466fdd6bd0937ea6",
  ],
  "executorch": [
    "sha256": "54b45205d28dc770a1047b44532e1607c8858fbb267a5d7f6a70bcf5c6910949",
    "sha256" + debug: "95e433a0f7d1173e23ee21aa348ca691b5c8902100472c65bdd9489966a4b29b",
  ],
  "kernels_custom": [
    "sha256": "c99404558fc2b219050d394acb9304530e89d2bd8fb508cb65f1925c9eb5e8ae",
    "sha256" + debug: "280f9ae352b2dc4cba28fcd591548b8ee356d36213860214127c9a206298da51",
  ],
  "kernels_optimized": [
    "sha256": "15ccc8b3bec8f2cecc4ce4086648eeff0886c17cd532aa3b5f10873da7661dd4",
    "sha256" + debug: "f65a4def1498c7d35f425a6b2bc744e0031380ef1651f51878f26b157c1a00a9",
  ],
  "kernels_portable": [
    "sha256": "0bdd977927d099e1ba09550670c09bb1933c0b1fd3087ea75ba2ca87f09de940",
    "sha256" + debug: "20cd916e7c48add33f29af52e5968fcaadc44dc16030404c1414c8e2bf614591",
  ],
  "kernels_quantized": [
    "sha256": "92369303d20a46e63902a9a2a845595d35fa37f3c7a4c0173e34b9283df70064",
    "sha256" + debug: "207cf2893d7417f16013a040602d69e680fd9aca6474dcce2db3ac8b14cf4002",
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
