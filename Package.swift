// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250413"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "b096e748faa563ee2279a72260700726dc3c2304ed0740d90766d6bcbe5cbcf6",
    "sha256" + debug: "450dd56c0713280b6c16400604347212988404b75cba263d37f6b716e05559eb",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "15c50cbd32f9cad86b32672a0d23615a7e4072a0372ad58841f5bf1a0e0e2bf0",
    "sha256" + debug: "ecd1bca30e9b2200337b65e94044f2302e046069b5a4ee4a9b48fa1c175595f5",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "403edbe1ad3e4f60f6a5e97f89bd5e34f3d6ac505c7120a00eacff7413a02367",
    "sha256" + debug: "5ff621ed072946d684372f10d3a5b2cb8f29fc960eabd91f1092acacfa267bf3",
  ],
  "executorch": [
    "sha256": "0954fa1f3b7a3efc989cbd4dad6e0feadfb69653d707b8fde47cb2c1fd04f669",
    "sha256" + debug: "251cecb0f4c7463660539a19b02be6d5627dd3e9a7e28aa8cc7c8befb1800843",
  ],
  "kernels_custom": [
    "sha256": "6f24f4218fd90239173c605dd23abc805a3522c4abcf96226bb572f66ff25184",
    "sha256" + debug: "56a3c7806c14d5fb0eef6942f74c66e718e29c458f78abc5a68027f10ea9ec32",
  ],
  "kernels_optimized": [
    "sha256": "884dcccae429dd5a451653486111b0a4b1703f9cb304c0183a07c3329093302f",
    "sha256" + debug: "aac098a7110facb4458eb81fa2063ce135bdf6127644e7d96d68eab1b2108826",
  ],
  "kernels_portable": [
    "sha256": "7a7669570eba52c91c7dda3d0b57354f88fe647fa9f706991979c998fa30258a",
    "sha256" + debug: "5df0d73175d9377d486bb6e4fba3f66e01cfd132ca68b41375a577a85b4d8776",
  ],
  "kernels_quantized": [
    "sha256": "d155f6a878387f4cee57e6713dc5c16f06896f388e79ce2d42a8a1cd82e30126",
    "sha256" + debug: "c590238006df32b32c661ea9b8f448c9363692310851b9df1ce2b429b91cdb51",
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
