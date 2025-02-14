// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.5.0.20250214"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "14b0689a1c80cf35f0b809b9a76c9aa125597e885c404f0e017e29e691a40bb2",
    "sha256" + debug: "b3f166d48ff42f27fe170e99c34248c3c8a2a26cccb4679f18090e538f8044af",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "85794ca73009bfb1854085a215a789fa7b86cb68c1a9508c021ca3b5b486dc73",
    "sha256" + debug: "8b363311fd7541d86f6e7084ade175a41ff372161e174189dc89ab3964148619",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3ed247435824667122a3f70073c297702fa0fa0fac6e46f65477731bc920966d",
    "sha256" + debug: "e06bb91798e922a745618ae302df6ba1f820cd86bc8790dcfc711131fb36f7e7",
  ],
  "executorch": [
    "sha256": "638644e9a98ce60731637f56ddd68d313acd2012f361b3a49a4ba4d938e83a0c",
    "sha256" + debug: "0a3af040a18f351c02809a4122b565e344efdd5ea2818135a5dddf6df4fdc523",
  ],
  "kernels_custom": [
    "sha256": "3e1b6aa7986455519db583f64e2847b871a09eca533b991aca219936f95cda43",
    "sha256" + debug: "7a4c76de8005f8e537a1898c23cd3b2ef32a3dd8e3afa79bda02fb5fa693c7ce",
  ],
  "kernels_optimized": [
    "sha256": "1ff39d540ab14473dff5311dab20d389e9d751cb1668a2fbe9621fe29b7b6ac7",
    "sha256" + debug: "04a2db0d1c41e4f6a351d3ede53ee8748d6630d6601b1f3206b54149693eaa01",
  ],
  "kernels_portable": [
    "sha256": "85c1459cb5f7126d2e4f0c76650ec7a62cd0ea2f25be831d379e09a58aa907d7",
    "sha256" + debug: "7b61a5e4eb23f4ee0379d5e9e71a283a1e644f110462b35bf0dcb1593bcd16fb",
  ],
  "kernels_quantized": [
    "sha256": "b2bf69fadc10b7117d896d81f5b54d4be8681296055febe22ea9f7ed2e1d79ff",
    "sha256" + debug: "1b3e40bab4edde980ac0b6dbb25f95acb18b0cbb1439c588586d94f1a99eabb9",
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
