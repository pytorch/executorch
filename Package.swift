// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250512"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "4ad8bd93351130e5eb8b36413c7f8c33ebff6d90ff7d4d4e2659848b7eaf490e",
    "sha256" + debug: "23fdc92ce38a66d755b00b4e98820a2d672471e4b5dedd128da484e20926dfc4",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "6fc24aa623e4ee5012061d7aa1e83e73817420663b3ed91af1b2caa6fa75323c",
    "sha256" + debug: "f031c5c44a3df79e436ab6c7a480038dd4d80f195a301d4611d5e41f56e2eccc",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "fcd28e59f1ef96f3bf6285a86ba8c35b8446a9f80c17a7386db8221031213e67",
    "sha256" + debug: "cca1a8d769c4a30da9a767a8cf6db3862f6d6e41995a999efd82624046af574a",
  ],
  "executorch": [
    "sha256": "7588d62ad0265f7e0a2762172b3eb5e0eb881b51d6ed4c74c05e5da231b85200",
    "sha256" + debug: "01378beb004fdb9f3e077c155a842a1eea8b84326406ee60009cb7ff2b7cdf18",
  ],
  "kernels_custom": [
    "sha256": "1328b69f1cd1095ad3aa51ca52de3df3b67641f0bfde0b011b4532a461ed5cff",
    "sha256" + debug: "f0b7528b245ebaacea846c2d14cde8020ca8e8023ded67398a016a0907086c88",
  ],
  "kernels_optimized": [
    "sha256": "30642dccd9a1317ba138c97f5a584b397b07be8f27888858e2fb1e73bf811db2",
    "sha256" + debug: "7f0194f1ed5148e72c37613312c3babe38e8a26cb2f014fa7271453df8c566ae",
  ],
  "kernels_portable": [
    "sha256": "7cc1412a51ebe6578b52957b2d4c5a5f576efa99d331340737c642555b8c3d24",
    "sha256" + debug: "a6a5b761b101971a9eacf0bc280323b3856f34c27caf078b2dd0882953115b02",
  ],
  "kernels_quantized": [
    "sha256": "0441c63699968e754984abfb8d31352639907505c2ab4ef0cf497853dad2bd9f",
    "sha256" + debug: "890cc9f46ae41ecfcde625b40c0203842778036583e4deaf72b1b05561b34288",
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
