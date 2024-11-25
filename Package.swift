// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241125"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "20f9e9d059ec41b8746a8bdfab51c95bcb2ea7932f10b066c340fd7851aa7b89",
    "sha256" + debug: "ad0095f42e3ab0568d6276fbabb4eec6420649ca84d27abbf3ec29befedc484f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1a04b0bd8cd6b1b8c32070c8d9fbf3b0232843cce394008341a78fb1d403ed0b",
    "sha256" + debug: "86c2866286b71c8404805729c2fcc37adfd47398944a19ad9252bfdf766b6f66",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "4cca1209ee74afe737dad16d5cd3374364c4ee15e88b539226d0be3e7450e613",
    "sha256" + debug: "63e21a12a839b9a20d7a59dbd1fe7f2f58fe877c86484139d75aeb3256619668",
  ],
  "executorch": [
    "sha256": "eb839ce3149710fefd823987d2ccc7be24332cbfed091ae86a6cf9a12e0993e2",
    "sha256" + debug: "478852a35a6936019cf3a3a0607bfda5ad8eb2155e3d9d4e6a054687d5f76ab7",
  ],
  "kernels_custom": [
    "sha256": "4e82038cefd90bdfe17320d1588ef527e01d4d94963a1449fbcf0986d274cd9a",
    "sha256" + debug: "3a3a853187f696bc67a91dfc2d8b28aff67e21c945305eebdbcadf6012be5c1f",
  ],
  "kernels_optimized": [
    "sha256": "1a23b992853812971c61249a44961ec79da91d9cb7250794018589050f6378fe",
    "sha256" + debug: "674d85de7d3350a179ea9146222a6ed29eab5a6b7592c61a2a3a665d28f64709",
  ],
  "kernels_portable": [
    "sha256": "1aa4b3570f032eaebb68bc1360553d989835eed7fa4aa1819aca69cb639a722f",
    "sha256" + debug: "c7312e82cc7e8b74b4422b1f92e2d157a2995a4837571d9346874201c82fffe6",
  ],
  "kernels_quantized": [
    "sha256": "d9db4f70dca5c9021113fe298046c871410d766af35123217d20cf93ddf0b362",
    "sha256" + debug: "32b13d0f4782a3e3793dd5cc6769d7564d96ad0166c59c29beb1e33ec3fc4049",
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
