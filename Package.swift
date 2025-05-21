// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250521"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "3a1d19c01f6a6497416e1eecc0716eceff264ff936561b71c1f659fff6ad6912",
    "sha256" + debug: "39932198f75769548d40227579f0548519d7f06285fbd6ff8484cd2d10d938f4",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a3bda34b063e47cdce8f7a11eec9d9a22c067c39f9cb98964485f3ae3276d3a3",
    "sha256" + debug: "40689b4522f6733ade82496505309ea0290a1791a698ad10342ee615950a24e0",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "740197be03a3a9f7dc5ecd0c82b0188d9584f5ceeb0053bd0c15f81628ff6cbe",
    "sha256" + debug: "cc84d2e7977a7ad2cb30366c64fc3bb0ae32d43025aac29ad031006f4e153730",
  ],
  "executorch": [
    "sha256": "d879a73575da8e386879df6428708b35268ea74ba33adf66a53a9d343234e652",
    "sha256" + debug: "d84fd8ddaf6b65c068c552f354f95ba3e3f5fe8c9835a25d3350fd79475c7611",
  ],
  "kernels_custom": [
    "sha256": "30ab760589bcd07e8548beaa1ef2cf0441923748174a8e5147a3009d2819024e",
    "sha256" + debug: "c686b87b9b79bd07b3a714b556cde840073a4787c632ac2fcc45b1242719d780",
  ],
  "kernels_optimized": [
    "sha256": "cac2574b90f08fd488783a7c928bd67f617637b7f288d3c98fe98f97dc8feb65",
    "sha256" + debug: "bec8a0d6f0e24f50c01da69357e824f2e05dec25afc399014749d1eae66061ba",
  ],
  "kernels_portable": [
    "sha256": "12f74ae295895eeec5623899c4c7fdf8df0dbea20df443efd5e95e12fa12943f",
    "sha256" + debug: "4a5886a61370814ba9283a708a12d2f92c11f372d3782b9f619f8c41c2c101a7",
  ],
  "kernels_quantized": [
    "sha256": "6bee20be63897e038e1529f5e7cbb0ad255efdde433482eb0e8538c2d15f6471",
    "sha256" + debug: "469d70e4f29d77c518396b3180b86e7c7582b992b6c64932c1785b578b58a491",
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
