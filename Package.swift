// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250419"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "55fbb220d89f576b9d38135814cf989f3ea0711d2ce5ca47d2cdb8273fcfe91d",
    "sha256" + debug: "64a9d8f7d8b0e42668a587f176fed758e6f56dfd6d67c251803ae2111d85dcfe",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "35f2fac03772af6fb82b7c39becf5b0c8952879ea06949763b06b931d0f54099",
    "sha256" + debug: "5cba48ad87b0ac9fc90e944265e6dde585578ce2d712d28a43a969ce935ed99e",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6fa786ab7b3192f15a1c949504093118219784ad1dbfe955109f09a845bd2ef2",
    "sha256" + debug: "f8950d244560fb8a1a7338bb88d75f4edc43fb9e3ec611a4e90b04731e1249a5",
  ],
  "executorch": [
    "sha256": "b30565941003ab44a07ec819e4dd2e7166904850f621e71c906abbd3b8ecbdbf",
    "sha256" + debug: "f941acae06ba4896c5df3f268a978d9c1b576b2aa7830e4326a70701bc4cab46",
  ],
  "kernels_custom": [
    "sha256": "519059a53aaf074f3bfbbbcb03de870798e4ff838b709ac55489ad87a8b86fce",
    "sha256" + debug: "9fbfe9ab93ebbd77d637badbc3cedda8858b12ef32f1e3c0069a55180972f98c",
  ],
  "kernels_optimized": [
    "sha256": "710b16857390ca1061a8b02da51dc052e1e7dea609f82f64bbcdf5cd2905960d",
    "sha256" + debug: "1901564a259adb2776d9000a3877301d1878dbeb5593dfcfcbbe9c8584e59793",
  ],
  "kernels_portable": [
    "sha256": "495f6e0f98fa012109343759a04151693576d6f26866ea695ea81f0dcfa9f2da",
    "sha256" + debug: "1c75c50a5bf986b8e7435d5e5a0c78291fe4246647a11c243e45983ba6811d69",
  ],
  "kernels_quantized": [
    "sha256": "ffb828af1290d271239f6b3838ca0629d0c60078de627baade7726eb7ce5bc8c",
    "sha256" + debug: "062379c7da67430f218f0f34a2fa0d1b24106412e7f8754a9e676411256fd520",
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
