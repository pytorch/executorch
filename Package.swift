// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250527"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "b7c9fd7006a0c8b496850d8d5c7d536d6f93f7f3a3cf10d2bc4759f11d2ebf9b",
    "sha256" + debug: "ea7660a97de94bc0d9f2f438e2d235d87bf97da0981d5dc441e24cb3612dc7de",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "229791a87241d95f007114b7b183cb5e7a284b7bcd1d6e715d4128db3857b651",
    "sha256" + debug: "27ebce888ff14803743ae85c199ec316fdea1831015045ead4d6644ed75bbc63",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ec854490927f08f1194da72469febb8d9a713abeb7a20a090fc58d04553a0c2e",
    "sha256" + debug: "d6ca5350f3419559b2ed724b00f2d487bb1c413f483cf99c5179a7d2225631ed",
  ],
  "executorch": [
    "sha256": "d41692dc50412172a41384deb593f128dcb20586e1c070636dc0483568f6c30d",
    "sha256" + debug: "9ea50b2b98dc6f11dcb6d91d6b973fb10aa806029efa27f7312a8ec87b2b04e8",
  ],
  "kernels_custom": [
    "sha256": "305de3f5ee60abebffa804caea23cfd9508774d120973fea1bab559e15ba9d2f",
    "sha256" + debug: "82ab80e5345786a8a0e4d3d62c03fcd1ab640c7fcb81e70110b65f8002a6a653",
  ],
  "kernels_optimized": [
    "sha256": "7b9dc3608392c9dc82020bc38ce5281e154c077cf7d923cf1e8ac0d912de1916",
    "sha256" + debug: "937d3dcaae58bd6022e767d3b7cab358264f6c61b5a327ef42ff65f027f685f2",
  ],
  "kernels_portable": [
    "sha256": "71c96f3664fd53002c1342002f3d4ec86497652769c759024cbcc0451b8c825c",
    "sha256" + debug: "9a16b4aa5c83f17f10a2f48c129f5c5eeded943f61fb07bf8f1049ee017cfd07",
  ],
  "kernels_quantized": [
    "sha256": "377ca490df0ede6fb1b30cadb7a335b306e2e70a45b8742991b49ceef0b14d09",
    "sha256" + debug: "10040a122b8afe65b5cc6f84fb79e0f5334ccd549b6948c3577bf27176fe0587",
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
