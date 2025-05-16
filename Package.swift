// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250516"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "351a4afbf74c04278107c5721005a53feb4068f1c28d69428c855c66e0dd28ce",
    "sha256" + debug: "a409d1139c4c77bb6fe9a71d0e5ed6a42c899cab0f28ea873f407299ea14061d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "696a6dfa846b183e5e52b8d8f6907b795849cc81fe25e64bfff98da8bc705604",
    "sha256" + debug: "b10494bc0b331d2241e1cacf05ad6e749eb76454162d1f3edbd24a598bc38964",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "db24b92102a052bcbb71914bdf190d1b7dbf82acc5cb13b45e49da52a5b376bc",
    "sha256" + debug: "370e5b76a0c9ed6ce1568904f03df7e2377e832906356013f3402b5b02061bec",
  ],
  "executorch": [
    "sha256": "dc5fcd6cfba4fb8c3ddeb95021a940afc7a62bce754e025d413a1af71570bc4a",
    "sha256" + debug: "7a418607f829b318f541b71803f27a158dd8334421e830cd2939df9e240246cc",
  ],
  "kernels_custom": [
    "sha256": "387067e142ad8a29a15eb9cd01b4edcffc0e4074cc55dd85a202e6f0019e11e9",
    "sha256" + debug: "0505d846a1c53951c30d9c743ab48be6f462120ff4b631649173a7e2fa5a9551",
  ],
  "kernels_optimized": [
    "sha256": "5f2cf6f1272bc122c0c4c6a0796035294e3199642ea60b52a7f39d8253fb0e73",
    "sha256" + debug: "9ea43214f74bf6177d61fbaf5305ba539e65174fe96ade22d4b8e1b1b807006b",
  ],
  "kernels_portable": [
    "sha256": "2dcfcfbcd538ff480781631134e56075d44f6ce887ecdda4840623f6da3aee64",
    "sha256" + debug: "d9a55fe2cf595cd82254ee8a68ef794651335e492f8a7fdad4dae088adf02be6",
  ],
  "kernels_quantized": [
    "sha256": "8de8e6bb219c3a78166388b86c73350137b84ac7502e36153ee1ecc99b7431f0",
    "sha256" + debug: "3c6ffb34e0484756e93db08128997e2208bb916f411f76c2a37879d93ee8a69a",
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
