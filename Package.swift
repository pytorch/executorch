// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "latest"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug = "_debug"
let deliverables = [
  "backend_coreml": [
    "sha256": "d9b44ca420ead4d31cc2df4353b87a347f24e5d94020be536cd34b7526cf2dc9",
    "sha256" + debug: "1290082422d73b62182b17edf0fe483beab1d4e32bc54d1951e2e5de3044102b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1eddfb381e6cf94dfdf21d0cd3b21b319fa055f369a23a1891497b14720cfabc",
    "sha256" + debug: "d07b7ddc76f2dff81ca984cf59aae666b3e1595c987b90cad004ab893eececb0",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "50e444502c2cadbb6124d0bda94ca18e6fbc969adcd0cd258d45a4a56b3a6ac7",
    "sha256" + debug: "5af97c52da4a58fc212e9be35ee3a8c13cd68fa3774ad1da9918da3e75cde3ff",
  ],
  "executorch": [
    "sha256": "60a71663b47b95422c8834882e59d4d2f469c0faca7463106ae8a58bca04293c",
    "sha256" + debug: "93d83a67a7f0d89bdecc196d3ac94079ff78e20bc2b92d4fb1d2bf70ab5dfb38",
  ],
  "kernels_custom": [
    "sha256": "da0c1e59996516370aea00f51282e493f5581caa53215b1c02acc77839d48725",
    "sha256" + debug: "f1f343aa9d8877572d0bd639d3e9c1029a463912434363a27e4ddfee2844f4d7",
  ],
  "kernels_optimized": [
    "sha256": "090927d835a2ee6a51e9312d155257d2643baa9115246fc29bb6c538a854381e",
    "sha256" + debug: "c982cb9b3db2f52c6fb5441e992b8b165d66fb7734a3163aa8e99f6be16158e7",
  ],
  "kernels_portable": [
    "sha256": "b9e4caf9cd75bf0ab605d60ebe4c3de7212b6f7dce72b890b7a9bcc268f838c0",
    "sha256" + debug: "a746c1b84e6d5f25b97f57877768b0398d262431d398bf7c8d63694097cb6ef0",
  ],
  "kernels_quantized": [
    "sha256": "3f338f2ecd20940fb1304349cf26802de2e0ba6058fad9820c98b5404bf373cb",
    "sha256" + debug: "a63d2eda7b247caf63320c72c44c171a34c36062a7d0c3fc4c3492008f0fc071",
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
    .iOS(.v15),
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
