// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250518"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "6e537f1b78255b6e2c5d39698a80d12daff58a7138fd79e4b2e8194e0a182604",
    "sha256" + debug: "71966ff62040f3de5bfc73f3a28868800d94bd4a2b953f21019f6f1d54f5092c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "362b13ebb920487f47b229db61d479f18d3b1ed2367b8c7cdd43fecbf166854f",
    "sha256" + debug: "e9b12a59056cb2c7a81dfc269e3a9ee35deb2ae638362cd74045b9331ad45908",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "cb3db43e2e39fe12ba4bd79ccd1ed196a422dcc7ea3991142a622eed6cf47268",
    "sha256" + debug: "45731ba58a9abd4f171933ac8cede36fd95dbbbbd02f9adb6e8e4838923cfcc1",
  ],
  "executorch": [
    "sha256": "c4a836aa9f91b520ed09d4c3d0688ea06e5629c3eb459c9224ac306af6da2bb2",
    "sha256" + debug: "9d2fa64d4423e898eef8fcece2293150504ea1a4af8552c2c254f8a75e9584f1",
  ],
  "kernels_custom": [
    "sha256": "75ceae3a7926c5ecd7854c612f1e61026979d90eac62d81da8548f084084cd3c",
    "sha256" + debug: "2dc354b1b812fe31e9572397411b05cde6e69e5da96374d88bdd6e43fe1e4310",
  ],
  "kernels_optimized": [
    "sha256": "9e5003b820d37a40527a129a8b1947e29b5b463431686e5e73ad61aabdcc1422",
    "sha256" + debug: "511bc8764a26a3de78b24dbf641cae00664c99fb282de3151167cbf1913a1982",
  ],
  "kernels_portable": [
    "sha256": "fc24ba7b010f23b8e278adb3ba7cb0adcdca51edeb72cc17270d8a852d626e48",
    "sha256" + debug: "314188f028f7f73f56d3632bf43ddfc93a59fc374534d49047b48c3c71f74375",
  ],
  "kernels_quantized": [
    "sha256": "fda25ab2fcf06ac48decf9191f9fb195199d7998de74ade6ade9bf5a3ceb5aa8",
    "sha256" + debug: "1ff9eb13c8a57765c10d7272c5da885bac5f059f6b7772350693797f11566e83",
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
