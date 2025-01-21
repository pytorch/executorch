// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20250121"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "76635b144dd9a1a0424e0758adcb8dfffb17f26e9eb999d89e88ba75b96dfca5",
    "sha256" + debug: "97ad7e555f3e598be9885d02fc6825998c5c602f4e26b48eba6a7bd7d96844f5",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1ac29c186bceba472f90b7526371150b34758eb6a1e5e15766f763fe86207516",
    "sha256" + debug: "c7cc8d1b137fb88cbcf9914ef3083797837a8a9326fe0bc0ab33f1dd46816e23",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a2f9bd845000b4f64febc70948348c8184fff5539a423b10e987322d4fbe8da9",
    "sha256" + debug: "5c5381c371c2534c49466c718181b41351491a8c49f254375755b060565b32db",
  ],
  "executorch": [
    "sha256": "b7bd1c96a7cc92591b43ea598a95d16049b39f830c7de991ad2cd1f9f975cada",
    "sha256" + debug: "5c29463bbf6c2863fee55843672147f68bb5642a90079716197257e2e61c472c",
  ],
  "kernels_custom": [
    "sha256": "166f1139c5cfe68becc974ad0b4a0dd8171699f748fafb8d6378bb66e6086007",
    "sha256" + debug: "508c3d3c09a96ec513b5036ce26a3a1f98930c42094542cec5b8abb232c5b635",
  ],
  "kernels_optimized": [
    "sha256": "87a010c5ca40df85cc9d6eb54ddd575eff47fdbe11bf2096b61e7be7ac3c8bae",
    "sha256" + debug: "ef881b48a822aae395613c243f9b7f0b4dd1f0e10ae4165184d02a26d78f6896",
  ],
  "kernels_portable": [
    "sha256": "b2b2ec4b7b0079333590e05fbf8d3bad6a5569bdd931d2789302ba1209a76503",
    "sha256" + debug: "e3f488e350306a2305fe66051a1a0869614a045e6b40f83dc64c68df8a396fe6",
  ],
  "kernels_quantized": [
    "sha256": "bd4a5ab47576f5246c563f27ea4887cb1907065397aea053583d54ed556cc676",
    "sha256" + debug: "0e379c104150d4686480215bab1d2704377e5f44a3560a2e49cc7dd1b7074f90",
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
