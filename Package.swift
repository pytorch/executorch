// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251231"
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
    "sha256": "31814687b83c6d3015e18d46647e70bc5e78108003070b499e9ec267297a28f9",
    "sha256" + debug_suffix: "1af6849c8639826e25be728fa311c1009d1992391190c1a7f1025677699b150a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8067e7776d7881d76a802daefeecd6a55f2ef8fe341b90bfbda45a7c0e1cf6c1",
    "sha256" + debug_suffix: "674be37ffd637b56e129ff05d7051eec2ca5120bea56634aedd6887837eee4a1",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "abb04e4f9cc7030b69de67eddf0def210d0dd62f2d0c81a01af7b16fcf5ea14d",
    "sha256" + debug_suffix: "93953191ec6dbcbba345e32bb80e9803ab73965e24f652caf2d38bafb49d9c57",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "7e74273eb2bd439476ff7cd3eb8b6038f82f9e5c8f72cbd62a3b71afc5a16019",
    "sha256" + debug_suffix: "a33d5902cfc3016108b306ec29cd61a992aec64219bba47edda68fcf9873ee4a",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "56db98f721a1a62aa9bb065b14f9ed344205f334fb609afab9c97192bd4ba608",
    "sha256" + debug_suffix: "741fc84078200deed5fdad3abcc7031b1a7745ec021faa326a0fc6fcc5805caf",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "fa69324213d4a97aab6763aed13c084b6377dd101642c7cb9f76333fc0f4089b",
    "sha256" + debug_suffix: "ab002779d50380e8d042dead916dc8b292a76425856dffcedc176c3624290cc7",
  ],
  "kernels_optimized": [
    "sha256": "8a187813c03a8d1052fc8471a604a4bb9365fe62816aa825fdd0691632f79954",
    "sha256" + debug_suffix: "78c0848adaaedd3ee441cc52b4fe54dae7a58a2b069cacc0b5a09bb476ec987b",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "3a888b1c0694dd47f03f9f982d129353686b37abe4c8b3bfb5531bd17381e594",
    "sha256" + debug_suffix: "62e9bd70cb5b1c01b4980bf53288484ec638d562b27970475bda9106c4933919",
  ],
  "kernels_torchao": [
    "sha256": "2c06a68fee445d0e0ef4ecd5a6c1093b63303e6735e6c88fbfdb2356903d2692",
    "sha256" + debug_suffix: "e1f1d7841f503a38ca96907589a2e01815c1b4474af504a3ef539d46a2c65c22",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "92edda9220daa1a8fecbc0e3f0cf68f3575da96ebcf10283593f334832c3d076",
    "sha256" + debug_suffix: "2f15bd615eb252a7b1384003957c237df19187773d19c0fdef45e1482fcc8534",
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
