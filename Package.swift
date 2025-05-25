// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250525"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "076aa4176342f951054b2e7307d61840d05c54c9afddbea7e5fd9a151611f19f",
    "sha256" + debug: "dcacd7e08ec1a3ad93b0c6d0a2103fd51093343e3c88e712aec67fd319ed6d90",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "299243fe31cab178e0fef1ca7273ed998086ff9740d42de6c97b57d1ff088465",
    "sha256" + debug: "c9efb445ba4e924c385189a9b51e67e6b0ed2edcf7dacee5ea0235c1ed359441",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "68d26c0f2abc5cf1c9af996aee8c0762794a838c5e869f5ea775707542139b06",
    "sha256" + debug: "6977549d282f28960dbc44a28bc566fdbf5e097e606568d1f44a2d9b040744d4",
  ],
  "executorch": [
    "sha256": "f799225f6614bf92a27fa6d904e560b1bc395736a676d25998ad0831aac8ec23",
    "sha256" + debug: "0276b081fc212a4bf719afe21b180980400b3988a07ead511e1cae907d5efdca",
  ],
  "kernels_custom": [
    "sha256": "5feb006cae6f409b986ddd20ef2eb066cdd863d76dc72670a4f6301de08494a0",
    "sha256" + debug: "f703addbe4943ec942adb0373fa6add28cfba7eadfaee5b2cf00954207a01d23",
  ],
  "kernels_optimized": [
    "sha256": "a359672f2885e68604af40672e8090f2ca4441896eaa5760f69366640e57b67a",
    "sha256" + debug: "79858ad01484071b95bc39b437ffd27aa480ed397ed21914d2243d6e3f1dee09",
  ],
  "kernels_portable": [
    "sha256": "9c23de3669edbbffd1a962d801cb77ecee04a1eaa174f34a82d751d10db8180f",
    "sha256" + debug: "f6a5686b001054c5feb1129779d0b5003f415cbccb737e1bbf6798b62ba53bdc",
  ],
  "kernels_quantized": [
    "sha256": "41d8a89ec4514df8c66197c1e4b8da2c44c4f89a6c846157ac8fa12783ff908b",
    "sha256" + debug: "4d38e3ca09a551f24fdd89c66c33038d7507822156e6c765c82bd18ea8f46f07",
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
