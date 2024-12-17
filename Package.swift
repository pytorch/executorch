// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.4.0.20241217"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "edc3e59b84dca12a2ee4ff41abb52727b1f7347993a365f8642ca65bb2d6dca0",
    "sha256" + debug: "83b1e9977b84bd25819e6a7031d2291e3633c78e336f6f546d89ecd46cdc1e0b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "4d3ef55738cdc1c577d01fc2703955cc19dec3f69990888a01f4d76903610ca6",
    "sha256" + debug: "0e5d46084f5247799e93cca3c50c8f6d56dde5ee952b1c0446989ffd42003d3c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1be15d9bd093114dd3b5f1db52fe300ca3b0b3d65fd8b057cd70e7f881e46027",
    "sha256" + debug: "f09309cadde5c73ecc64e27fb3e9b52c9e84c2ef3429cfb5c37a95294ebd3117",
  ],
  "executorch": [
    "sha256": "1104c0be029bba4b2a6396653475159d508b8c4369d36f92ed33ac65806add85",
    "sha256" + debug: "ca465401fe79f3d205f215e4598a459b940d7a22473d17c398d538ef97149e4f",
  ],
  "kernels_custom": [
    "sha256": "2c62868bc53e4fff07cad969b276f663824f9313df7452c1bce9de2d9217b874",
    "sha256" + debug: "85a2be4b7c2d161f2e572047d152c5ba06906f07ec3c7ce913baf8886784f078",
  ],
  "kernels_optimized": [
    "sha256": "eb79c303c8cca9a404920a2e195ed16010cf085c802d1259a51fb3acf87e3168",
    "sha256" + debug: "26e1ff32ceee03095cd8970c506e6ddcba649b4c4e0c5ecd0f6801065a911774",
  ],
  "kernels_portable": [
    "sha256": "5c3a02aab77f282a139a1666ee1d275cf8c6c7483f2c50e0f83d811871d13dbe",
    "sha256" + debug: "270462b3d18e21e9d3230f1eda6b51a4a2c3e10f71c073bc1010451a889545ed",
  ],
  "kernels_quantized": [
    "sha256": "f15fa30daf23cf69b195d8e351d0f5c428241e19c5997fb1ad76c606c315d025",
    "sha256" + debug: "5719eb1706d841d6426072571472d8eeb04fc65cb60fb99427ad758021ffd9d7",
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
