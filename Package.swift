// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260601"
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
    "sha256": "82847169c8b39c243da992776b19eaecc34585fcfe31ba15310db4bdea2cd9f6",
    "sha256" + debug_suffix: "d6760231c12b3cef23b86c42a00c7d76928847b3e8c0c3adb309f601adcf845a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "72a20ca582ba9bac6e63807d0568678d37980d7f386f58bab2782870600cc487",
    "sha256" + debug_suffix: "d3c83a0b061cdae6f640fb31ee9ebbbc73bbe46f76bb1f6afb270bceb99f1c64",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "040f883f9162764a7f565a3ee1e4b7ea27d077761ebc833aa7c177ea18d80f4f",
    "sha256" + debug_suffix: "6a1be332e88d7778376d26ecbfd3d83f2135529e8224cfe74a288c70605b299e",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "c366123e7d63c3a0d2ca2ab1b4c2748f5c1d70ce34103b4569e33ad2d41a4bc5",
    "sha256" + debug_suffix: "ecab385748938a0431f551c973f5b29ad7042e24625a62bfff15c35e48dda0c8",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "41887b2451296a5386e673739c793b7f5d8786510e7f85beaa10bb78e4384491",
    "sha256" + debug_suffix: "f79f081333fc7082b6bff88da446c1b3a9f3922e9471ec3a899ab0e46c1eca56",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "57d52e4aa47c6acaa282a571c8b5d2ca5cfc9875498caae48f36220a86508118",
    "sha256" + debug_suffix: "8f73eade6e9645e9b960d3e06b174dfa137d29238b745e218d867f2fb78a58ea",
  ],
  "kernels_optimized": [
    "sha256": "28e4f7bac33875caf4a6ad5d2343935f444fed174a941f527037213ddf99cd88",
    "sha256" + debug_suffix: "6e11c585af517d38f2918198427dd52dde4704aa41e9a9fe78551c13e1168863",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "2d7b1b3d4b14d17ebdf53c6d649701f865214ab22a5d8a1ee97b88c3ddeac6e6",
    "sha256" + debug_suffix: "da4dacea785c8015b0c56c7e5de45a7e1e5f0d3e47af184870a8580835dca18d",
  ],
  "kernels_torchao": [
    "sha256": "4d1b5088175b77fa6da9dda13acbc0e7d2693e37411f1d54424e490d679fb3de",
    "sha256" + debug_suffix: "2b98ded9fc5f0f2240f1196d2759979b13144f7d3d62dc1959eb50166512ca38",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "7e3638893a8b1a4bcb1af1fceae075109c418eb6d518754d2ac18d87f92b6ba6",
    "sha256" + debug_suffix: "e9508f7ca7313901de3c89762350552589947b6097076c11687a6f8de43b3dd0",
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
