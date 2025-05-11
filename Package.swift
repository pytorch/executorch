// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250511"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "73ea8fb337279a1d2ef5259febab9a3dea70157bb2387ce01910c11fb3644a62",
    "sha256" + debug: "db8c967c8d55f6d4fde9b704cb8c8c1734c48ce5f2ab486ea2ec0c52f031160e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "783f539df7c67d63374ee74f128f3b24be48466e8d159d5b6a790fdfec04b368",
    "sha256" + debug: "6bf1b545f8e78c55f64af197db660fab04a5ca92d975340f5954d3289fcf6c2f",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "2107d5aca63498e4ef023859f04de87d12f0dd2654df77274184ad8b0f611859",
    "sha256" + debug: "b53fb828d6304c631a829d74e84fc3f5d3005b9a66dd077ddb90dbe5108c9de7",
  ],
  "executorch": [
    "sha256": "0ee3751d12b8a6f139656d33ccf6ddc83424baab53609dc9d1ddc41ebef05614",
    "sha256" + debug: "78354eff864345c1c9e0d0594bd6d2e35afa439c415c48ce9c0ddc54783cfb36",
  ],
  "kernels_custom": [
    "sha256": "03ceb58dceb0a24d59d8b4994c7b932d814b0986201a73f7f0a535b51e9a56db",
    "sha256" + debug: "5ae11a5766001f12486384a2703d6a97d5b2368217bc5d3ddffb311433d1f04d",
  ],
  "kernels_optimized": [
    "sha256": "e2235020a566ae40da3f2c394ffc6a22073c5edb194715561d419d936fda3602",
    "sha256" + debug: "261453b027a97bf10fa914bb86347027077ca4aeb1d8691539c665db268d6a54",
  ],
  "kernels_portable": [
    "sha256": "5920f6de79060ca1479cf5bf78aa57f68c4ffbcc671ea2d07bbce17d8f3ca7f2",
    "sha256" + debug: "3b64b823a21dfef8e00f565ed3bd812669d253aa07a1cfc2af487d06ae61d661",
  ],
  "kernels_quantized": [
    "sha256": "99d2dfabca8251e8a684f5043b6978663edd98d2e490e6f3ae1800d83561a30a",
    "sha256" + debug: "3dfdbc1c0391a3c0d36ad6612775984f42cfec0e6e7837d59552a62cfda145f3",
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
