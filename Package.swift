// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250218"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "bc3c7f0af7fb5021c0332e4beb3cfd2d19550870cbe61503fc5fb0aa780b31c3",
    "sha256" + debug: "4d8fe5b143b23fe4cb45da67545722718dc0a24b06fdd0ef061ee0459f6ecc5e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "60dfc38d7e44deaf216939850a29860aee01a21ba337293cac57064f3e578ccd",
    "sha256" + debug: "c53b26ddb34464ece40df5d47270d17ed7d90a404dda3ef255ed4c1708571c7e",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "05247061c8cf15edafc262bbd0c9c07877697e7b6db050e35966466b9e730be2",
    "sha256" + debug: "c54630d044c66fc4696b3f77ab2900a6327040faee2860b4548e67e387dde815",
  ],
  "executorch": [
    "sha256": "052fa8a5726dc54768af50b775f2edf2ef3a1aa934d24fb2b57a9c0fd2bc2739",
    "sha256" + debug: "7c3f14d700ca16c15a64c7a2035bf2f2de75daaf9195941b3007fee0919e628b",
  ],
  "kernels_custom": [
    "sha256": "1a7a753d0b17ea6e2ef7632ccd5d10e41dd8f5cdffe624f5c0cbe88a5e2a7527",
    "sha256" + debug: "658a4925757969d145c7a5c37d4599bbbec04940cf8db4d9275a57be5107fa7c",
  ],
  "kernels_optimized": [
    "sha256": "b0258362b789d4c9a275ab6d91c246a6236e6c3e3853946bcde1c9d9df03722e",
    "sha256" + debug: "db709ed1cf7b5ba75820a37e2ffa5e4e904b2d02aac0ffda11150e6586e2e7c1",
  ],
  "kernels_portable": [
    "sha256": "e88d53ec73186b45f31414c02b6271dfa8723d4ec9c33620338d160489c5fdae",
    "sha256" + debug: "bf3902a6d7352f80fd17ca4400ba190ff3ec453aded6de6adea86e26d7deeb23",
  ],
  "kernels_quantized": [
    "sha256": "054ffd139f5c7258d3cb451e6e2a18b9ff8dad7fe581205a61f1c2e169d5df7e",
    "sha256" + debug: "77ec291c86ce01828fb6b6bc0c2a4cb2e81a029a8526685ef4ba75fba06ee00d",
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
        path: ".swift/\(key)",
        linkerSettings:
          (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
          (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
      ),
    ]
  }
)
