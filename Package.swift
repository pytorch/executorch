// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260405"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug_suffix = "_debug"
let dependencies_suffix = "_with_dependencies"

func deliverables(_ dict: [String: [String: Any]]) -> [String: [String: Any]] {
  dict
    .reduce(into: [String: [String: Any]]()) { result, pair in
      let (key, value) = pair
      result[key] = value
      result[key + debug_suffix] = value
    }
    .reduce(into: [String: [String: Any]]()) { result, pair in
      let (key, value) = pair
      var newValue = value
      if key.hasSuffix(debug_suffix) {
        for (k, v) in value where k.hasSuffix(debug_suffix) {
          let trimmed = String(k.dropLast(debug_suffix.count))
          newValue[trimmed] = v
        }
      }
      result[key] = newValue.filter { !$0.key.hasSuffix(debug_suffix) }
    }
}

let products = deliverables([
  "backend_coreml": [
    "sha256": "175b0cb1987fb8a911b71ed714953e8c44ff55da8963660e9508d67e929cfc8c",
    "sha256" + debug_suffix: "7e9aa2e5a902784df1d26e3cf0c7dce0694642fe28bd7f840e15fcef72e42b1d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7670f614d1816188962d9a5517111480d4aeea2cedd90e964dce531daae0ef0b",
    "sha256" + debug_suffix: "9520d465b754f1396a9a16ece139d65089d545343d29de7634a09d803b941a62",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b189521139ac92556a5d7ea839931fcdb9e33b4b9b02993431dc2fb999d575fa",
    "sha256" + debug_suffix: "cedfc00f635f0a3cb53cfbf92015d3cb2f9ba394feb873794ca4c304acddaf27",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "c06a2ee59de5302720777af822a1634802f2b3f88c6040c384869b374b2c0b69",
    "sha256" + debug_suffix: "32ebb8def71f580e278e1b47a21208cde531b647be08aec40b091d3c3a54dd59",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "190688992ed5230154b05c0f9fc8fa9c00f0c3eac306a7d58f50757582258268",
    "sha256" + debug_suffix: "b2bbb2821d2de195fa8ac75dc429f0ff151c11024547828b030d38228da47d2f",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c73b84988754e4a52a30506bd70fa6408fb693d409ec51ce37ffa64586d40220",
    "sha256" + debug_suffix: "a67c5cc440563e1369637ee3b01cc721630ef12cb8cd936938470131f39c4851",
  ],
  "kernels_optimized": [
    "sha256": "48ff735b58b2dec1b60ef80425e11e824025136eba5a47e6bb47f0028fe0f7d2",
    "sha256" + debug_suffix: "93375ee39071d17fa10550f080a1bae6cdc35f8536a1a2bc0d4ec27a97608501",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "f15873aaaeba331b26c9a109129abe7526a35f092b3084a88ffdec3a14634c28",
    "sha256" + debug_suffix: "4b7d68013b1e3959aa5e677b46251b0dc2ef0ff7bde218cd20776fe1ea515687",
  ],
  "kernels_torchao": [
    "sha256": "a989fdb525b1484923ee02563a2a7c0d0b87d3f71135b0a8e44e0465238f0d19",
    "sha256" + debug_suffix: "f639dbe8fdf3d3e242bdc07aa7317fe693de0ef63733b87c5a73ffa0a990e4ea",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "700fff43b9cf6c73bbb135af5dcc502aa21a796a24b55ff7fefc66bb799c5163",
    "sha256" + debug_suffix: "d2e3328951f81488035c71af226bc48c76739e6ef97e979d4ceb9009344a8ac3",
  ],
])

let packageProducts: [Product] = products.keys.map { key -> Product in
  .library(name: key, targets: ["\(key)\(dependencies_suffix)"])
}.sorted { $0.name < $1.name }

var packageTargets: [Target] = []

for (key, value) in targets {
  packageTargets.append(.binaryTarget(
    name: key,
    url: "\(url)\(key)-\(version).zip",
    checksum: value["sha256"] as? String ?? ""
  ))
}

for (key, value) in products {
  packageTargets.append(.binaryTarget(
    name: key,
    url: "\(url)\(key)-\(version).zip",
    checksum: value["sha256"] as? String ?? ""
  ))
  let target: Target = .target(
    name: "\(key)\(dependencies_suffix)",
    dependencies: ([key] + (value["targets"] as? [String] ?? []).map {
      key.hasSuffix(debug_suffix) ? $0 + debug_suffix : $0
    }).map { .target(name: $0) },
    path: ".Package.swift/\(key)",
    linkerSettings:
      (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
      (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
  )
  packageTargets.append(target)
}

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v12),
  ],
  products: packageProducts,
  targets: packageTargets
)
