// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250501"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "6193b6e3cacec2fbc19a61596882305b47ec7635f5aaba87c390c02f89500694",
    "sha256" + debug: "7f14f344016d849e53b76a6cd443ee23f241286750fda05bd2b7738d87089962",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "918bced72787864169f2b28273db203b140c04dec534f238fb899862d88da12c",
    "sha256" + debug: "57090b2aa8cf1f3e55d66c9ccbdb87c90e703f4e3e9b10176f04005d7f109eeb",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "154609b79cd2e76899ad93869c4b15a9c8020348caa17a9b66e47eb4b7d0815a",
    "sha256" + debug: "3e832917d197ee56e1de8470180510fe65e02f31551aa4975b6403e94b519b46",
  ],
  "executorch": [
    "sha256": "c63cf1a7b64a15f25de2fb566071f127557f18b2ee65e647ec00a748440347cd",
    "sha256" + debug: "951c663e101f1854a714da03c71ab53794a1307929b42a0af0f5524c9052bcb5",
  ],
  "kernels_custom": [
    "sha256": "b9e5eb6f50f4fc4e688892261ed4e9d558ffda5786872eb9d05ac1b3dd0ca6df",
    "sha256" + debug: "4e5f1a6fcce55fb646ca247509a7c4dce8694ffc0d7a905a33b7d85a3bc32f8b",
  ],
  "kernels_optimized": [
    "sha256": "74fa2955f3d5dbf55b9a40979e59290c08d536ee9d268eef2e75c556015daf3b",
    "sha256" + debug: "b0ea75454b01ca6634acd6a7429683f5cf8ceafa82e5080cb4a3b78102dfae7a",
  ],
  "kernels_portable": [
    "sha256": "2079c255cc0b60b2035efc84667b5903079836682c2e05b580ea52b32d03d1f3",
    "sha256" + debug: "8760829de4125931ce8c8bb829b0dbb67e6d9b97f7dd140a6082b03cbafe55d0",
  ],
  "kernels_quantized": [
    "sha256": "168d26f435eb84865695442c7539e2c2917153ce1a95d6823571f9eadb949476",
    "sha256" + debug: "c1cd1b7966a914cffa10eff443a87f916265fb01366a8697b460b1988c98559f",
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
