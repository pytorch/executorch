// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250131"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "6b84eed9ac30e423bc5207b47f48749a13df0133b39dc62dd369ecefeba8ca9d",
    "sha256" + debug: "443a6eb1547bff046199f319c7959217c146cf6444c30bbe1d55a2eb5062c920",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "6e8da570f60c3ff6eaee413f9195a707f6b88fa1aec88202071ecb98a7ddd490",
    "sha256" + debug: "3f008c12b4ece1d93e0c5de2f25d72a16dac5c04eb4e8deb9b03dc24d3f44d0a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7bfc6656472f6f0371b5026c65686835ec75d35f88ae055e82e2483305ce5b00",
    "sha256" + debug: "4f8f7938e2527818959a6cd5879d1e867b49bb22a95f421fb4bba26b4bc2f796",
  ],
  "executorch": [
    "sha256": "c8af6bd477493358bff45c973b4c7417c644bc5807b8dc5deda957ff4326c75f",
    "sha256" + debug: "e2fceb25b17a47c3ec361d6da43eaf273eb33c83f93272bd1901e376b70ffdc2",
  ],
  "kernels_custom": [
    "sha256": "7a5c6dda17fc956afdb1443d03a135d3281034702b29d81ad622dd19eab845c7",
    "sha256" + debug: "4044f66314375968c1050fc3518f32349ff85906192c42a4d0cbc82cb4b74b6b",
  ],
  "kernels_optimized": [
    "sha256": "b20a736bc93286c89440b8a5acbf50e34d6c8f5258894868a0ea42c8c76629b2",
    "sha256" + debug: "02e323499bab85240455bfe33897a5459669f9ed139548694fbfd3418a4aa362",
  ],
  "kernels_portable": [
    "sha256": "d3c1666d092ec1cc0b86ffb42c87014d3bf752aba14ed478409022b9f4e877bc",
    "sha256" + debug: "0481c8b414434d6ca7ed25c96172967859fd5fd46d8ca0960c981ff1c8388c57",
  ],
  "kernels_quantized": [
    "sha256": "60e3622374c26d5bedf72ba1d064ea785834aec7a51118454e9421e8798de117",
    "sha256" + debug: "4628de654e863e556d253ed484f61ffd095e175e27f14ed7e05fcd9f369caa11",
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
