// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250401"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "16b446958747fcadbcc16c359654a222c510ae7c7009d74b3e49787e3e405c8a",
    "sha256" + debug: "b5bcf258210ab731c131c188fef83d0d58c1f60198da611ce953d2585124d43e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1a7b5786a800cac705cd79cd46577675af1d61abcd334ee2fa3bd34a267625f2",
    "sha256" + debug: "efc20be4d6567219450894701620bd0fe79bacc99a2fe538783c6cf0ff3c8b27",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6196d0b3ed9cfb2863af51a3d81c0df6f5177833df46752f64b02c7bec4e7ef7",
    "sha256" + debug: "57f60c2251caf9e4f4f285109683f0397cf65e39bd2463668d422f28c4d99605",
  ],
  "executorch": [
    "sha256": "b111544d07fa3395146a95ec082298abf10dc8385e2c72da3dba6c56190bab7c",
    "sha256" + debug: "955aa10a1b8b796fb5a60586602bf7c32cc53d8ec7f0cd990ae7b0aac0e3076f",
  ],
  "kernels_custom": [
    "sha256": "a35555ade7d615fd32b79d531dc15d1e1876fbc33ebedafd9509e547b95eb592",
    "sha256" + debug: "914105835d7b71bba854c03eee856985a3719248e6e743afee86ebe2c56594bc",
  ],
  "kernels_optimized": [
    "sha256": "8fe93908a236ecee0e9eb42f85392e3dc30ff94c1fb3034391eb4c235d87eaa8",
    "sha256" + debug: "2f35410ee61b5efc0cd79a846197dd3ea07dba9fc2ead0aa80419dbd1508f0ea",
  ],
  "kernels_portable": [
    "sha256": "5dbe3bed49ef9ae529e67890ae459509f309911ebbf7842054c5581a36fb4ef3",
    "sha256" + debug: "9ca2f5cf32ccd973cefc502ebb6c689949614373dfaaddcb0f4f04dad388fbf0",
  ],
  "kernels_quantized": [
    "sha256": "d2e4056d95e6933aa7506f0a35e417724adb7dc7a32d1d4f829e5edfe624fb93",
    "sha256" + debug: "a2093ab804f8fe15813e2836de41d063ceb67d6e9f672c0a2c84171baf0f1846",
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
