// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "latest"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "coreml_backend": [
    "sha256": "9bf08789e26f4b348facc9b561b7e85a7dcacdb69ed734c11da8c8786e4ff98e",
    "sha256" + debug: "1aa05a93f9e16a61f6303b0990f18247313ca25f1738e75162690e9d63dad1e3",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "custom_backend": [
    "sha256": "dd14704b634c8a32e844a24ff0651b25f5eacb805150f18c308796d5cb977b1c",
    "sha256" + debug: "2a13deac066c1d110480b2cb7b87c724e7fcbd9c21ceca0b61a9d7917cfdb9bd",
  ],
  "executorch": [
    "sha256": "8ae82ac43822781aebff354ea9993df5d9aa55e61ff57867a692234e19f788fc",
    "sha256" + debug: "44f02f77e6bc1700ee9ade2ed2ef93ae55313b69824c0a890f0325f7abcaa4a0",
  ],
  "mps_backend": [
    "sha256": "97db0fd2b458ff4dae3f4e927d417b4ce88ef3bd4114759abe8372a05bac84ad",
    "sha256" + debug: "f22c373804caebb76c9ad810163e6eb2a92e4e95b98777f7258b14b8e57e9459",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "optimized_backend": [
    "sha256": "7ff98bf0b9ce540aef2aca7531ed6c663633d48de78183f49858ff885a7774ce",
    "sha256" + debug: "01c2b75de4a2c87141e69cd166fb186557b1e2fab8a05581c9f406260487aef8",
  ],
  "portable_backend": [
    "sha256": "d1f6a097011e46fce73f27b84ddf6ff3b6c63b8c77d15dd3e6c4e217e830fb28",
    "sha256" + debug: "b9b205e0ad7e9e0f814952d5a482cc35f798d610cb6ec2ef0f1c1e1f6b3a0878",
  ],
  "quantized_backend": [
    "sha256": "3743df5e0cfcf79faddb9c58e89c3280610548722d0cc569112e314b2872eeeb",
    "sha256" + debug: "00a17537440ada5a4bc715da067da40c8c695f69eb3f7b7cd6d7d9c0d8f368ff",
  ],
  "xnnpack_backend": [
    "sha256": "4b93d336ec6568534400ffbe3c03aefe0def86c05b21a4ae97d1517958a94270",
    "sha256" + debug: "77790ceaae9aa227a812f572e0116a5dd6fc25083c2795730162570df8c5a728",
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
    .iOS(.v15),
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
