// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241129"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "8b10194f8a5f21387f5bba11c56fb6ad073e292909971d54bf5172068f4e131b",
    "sha256" + debug: "3a4a0bbe808c083555fa2a15405abdb730350628dd28a706437d3d08ec71eebf",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7e9c7e6015c63e18c5f6ad5c94abc81983a47936d48a2a9496baaca0258a9941",
    "sha256" + debug: "c59254ca0e53510af423a8c4eebd6ac74499c5c36f480709a0efa3c1700dcc88",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a9cc8c1c2685403199f38114a247b308f8755cea08e3e4b59249012c9df1e64e",
    "sha256" + debug: "6efda282aa2cf45b0d702f1b6633d9c56a8167ebd09b35a319bd9a8d4fc4ce35",
  ],
  "executorch": [
    "sha256": "fbaa8bfde3e6490645db99a34459ce2308e2a28a025153d6667b2c810f6a25dc",
    "sha256" + debug: "113cdbfda8ad41fe29535a4cb3750baea2ca22152eef13067207aad785214a85",
  ],
  "kernels_custom": [
    "sha256": "edced251db0139d9aad98925c37062fb1bcc5e91dabd663aa722dbd1482a992b",
    "sha256" + debug: "dd690c6b0f9b0bf8267ff68cfea24c18df838573393ad66cbbbcf95f01d1b4b6",
  ],
  "kernels_optimized": [
    "sha256": "583f87eb08c661629d9a9ddd48278c4d4418869374225ec7e4b7d1a8fe19cb58",
    "sha256" + debug: "a872519bceacdd559fd57a2628d0ef9cfc62eb392a9af491aaccd7d87c8742cb",
  ],
  "kernels_portable": [
    "sha256": "1223cc1bb44f817b87d067b5215710af3874b788414c33a2d9d2930b23c4a57d",
    "sha256" + debug: "b519fa152815dc2d0a966d570da3918f25166f580acadea35e4994738f896b61",
  ],
  "kernels_quantized": [
    "sha256": "866ed74dfb66bccbc6c9d392d059400289d712a8111953fcaacd1f5328bc08e6",
    "sha256" + debug: "f280364a164fc917532857cb208ea92d729ce8d44dd6a0bb84545052c0cf3c9c",
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
