// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260112"
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
    "sha256": "7eb95082831658b9a9a75e04db11074004b5fcb2507cde027a80ed2c61d26ec3",
    "sha256" + debug_suffix: "a88c9edd49ab0315b7482358e41f0023205e1bfb6938ffd9d3a7439e9b720c57",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "fa2e59972761ff767cc46b4d25eefabfa441e1c89f70e7234bf7d5814b0b0022",
    "sha256" + debug_suffix: "41e938f8cc16f25e5002687c8df68d04bcb33f7324f32b5feeb47d9ba04a982a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a1d8f084ee2d97d34fc0cfa0d78227c738b41348e1dc913b61b8b5bd2f8a42c6",
    "sha256" + debug_suffix: "a4e512127aa241a531d987dd683fe9bdf666a432ccb79d55e014759b49e8f9d1",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "9433dfe00355e42bf5457d27e2b66c0f027f06caff910f395cef53e3e0c0a3a2",
    "sha256" + debug_suffix: "183329d0ca6f5857269a5d5456c7e6b22d24f8b65af340a3c27bcd20af35cbad",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "fce6b0dd36a7c6d34c615f042c4256333e83cec2fb4f99f50428d0e45917a324",
    "sha256" + debug_suffix: "18f0a1328a4e4520f155aa1a9663c79232673da996965059ff6f7ee9d260fa8e",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "34be4da902af6d6495e419bada65aa74cd6a98907ac34d5e77f7f77551c3098f",
    "sha256" + debug_suffix: "d68d2aeede879a2267d5271cd69b8ec2c092274e1c373fdc9ee38eae99f351fe",
  ],
  "kernels_optimized": [
    "sha256": "60ea50a50365ce545f63b66e5a85693e00d1716e72883a267865af8c6751eadb",
    "sha256" + debug_suffix: "b146fd53be4173ae757b7ef54cf5c191ca9c71b02ad87f3750d18bfc761dbe12",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "5a9f20072e83b4b8cf2d38fb655062d9f2ea4127411d0e525a1f251651f400ba",
    "sha256" + debug_suffix: "9d3ee5473af7e9d53da759a4eee83672f762aff21a82ce1d9818ce6fc041a029",
  ],
  "kernels_torchao": [
    "sha256": "5cd75783defdf057b50759b79724b599f70a48e16db60113e3c3ff5466e0913c",
    "sha256" + debug_suffix: "87e0336f4002761665d5eb9fcb78da970fb5c2486262f81f9af28e5074d888a8",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "dab69c4844cb1f36cf63fd9b1c365d40aa50b1c60855ee76e34fa367845b0ab4",
    "sha256" + debug_suffix: "c2b4a1eb57e72be9c302c970f8c6a353bfcf1a168c09fbbfa88e7f347a46d208",
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
