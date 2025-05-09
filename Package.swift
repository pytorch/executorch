// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250509"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "7b238a64b17dc75fa603ddc45260ccd8915f61133234c434e9c0171d05dc4cf9",
    "sha256" + debug: "4b8b6c716755ec208666bf9ff50d08ba54564cfcd8f2ab9d8b40b0e76a931833",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "308a54e4d0de45173244f28e6fea6e8c8796d8c0e525c4ce7183e4761882d796",
    "sha256" + debug: "034d50158d8eacb906b623e03bf5623a8b5fec85fd7a1cee5f2681b2a734c9fb",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "75677b0cc74e3712bdc044ab0b55722245a8bed90a08fda2c3c558cd236ffd57",
    "sha256" + debug: "5c64266311704b17f19e3f49281067654ffd63c0b7f7ec9c8ee8171c6d293f96",
  ],
  "executorch": [
    "sha256": "4e205dd027fb8e8ee562f35c5d16dd6814dbd7295de42d6d43da829a3fae01a1",
    "sha256" + debug: "0c486b58cf85f784214549cf336b21b669a2526b97bd418a9f4ee74e1e5a47c2",
  ],
  "kernels_custom": [
    "sha256": "5cd21e7d73f67588819934f47dc807103401352879a92da804b109f70cb9d9de",
    "sha256" + debug: "16857e870526bfe9aa2002d947f35ee7dd746b63dea4d4593890f46a280e1c58",
  ],
  "kernels_optimized": [
    "sha256": "c1a4366a7c24dfc342512b7071c0fcd071d168ee947ca5e51d73aad9a8faff52",
    "sha256" + debug: "0de3c16eb377d80c697984e2968dd00eb80b80374f56c1fe35bb9dbdf597d981",
  ],
  "kernels_portable": [
    "sha256": "b4155f707a4bb64513af3f070e75f414e50e6e7d43c106d6ee647c79bcfb4cbf",
    "sha256" + debug: "87822b4a510014f020ee5b669a572018e729377fddfaa7cde3d79f3ea30d7aec",
  ],
  "kernels_quantized": [
    "sha256": "52d933d912249c8f2c43fe4e4414607dc7a6bf8131f3d3726d3a2256f70e210b",
    "sha256" + debug: "e2470214aeda4fa12b37928aa86f3049672a6b668119aeaf706a33df928ec0f7",
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
