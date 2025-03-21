// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250321"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "e10c007155ca23bee2eb63a07093180656a2dd31a18005fc4ab22bab97694cad",
    "sha256" + debug: "cb428781ba217c1d18f9ab90e96c8a0d11e52cc18c84079248341ed8b96039a8",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "3b0e1914cc2c1e246d17bfe7042e8123e49ac1408761993773924bbd57fdd154",
    "sha256" + debug: "422083b031ea6df437d07b60a86baff18974aec49f8d6fea628a9ef04fb91c60",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6e98153ee1c8de1370abcde0f5f4a5ed5e63bf7e7333e9a3fc76c69d33338581",
    "sha256" + debug: "3bfb4309ab429c7a016e354ea363f701141e42a135852c807fa03d57bf4b9b50",
  ],
  "executorch": [
    "sha256": "1654c21c11d68a45d078fd24ddfafcc688b09846c81992d83e3ee23d088d3771",
    "sha256" + debug: "7f8ef174db924748173d56437d3f84402a92c9e560a7667c11bcff5f88d2db85",
  ],
  "kernels_custom": [
    "sha256": "46278f20f4137c806aee6f173ea69a27e541fce21edbd951708a53da36c846cf",
    "sha256" + debug: "0f0b1a5d55918ec6ea02dfc83243b9d7f0b07267d128640b9210163a6e0b27a0",
  ],
  "kernels_optimized": [
    "sha256": "c3a49dbdfce8098a53356634f8bfbbfeb21ae2f6dbe77e5f77d1b1e77ba6b39a",
    "sha256" + debug: "cb10f6ec6133c8d09250b9a3c171b2f78f67e027aa440deacab7bf104a361baf",
  ],
  "kernels_portable": [
    "sha256": "c818040fe6a168d251f66d9d5398cbb814727a4dc6c8b1f4201213a568c8d59e",
    "sha256" + debug: "31a97dceea6227c20d25d564936308d4469436abf1a350b6c9b11b73d4e6ee31",
  ],
  "kernels_quantized": [
    "sha256": "68b57a4fd7fd77103688d1e17883fd03f7c29d4fe849aa0260bd86cab91afb53",
    "sha256" + debug: "8b74023b7a0c3d777d98d6f2a102563f69eab3113c6ca00de83c80aa3eaed5c9",
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
