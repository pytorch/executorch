// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260109"
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
    "sha256": "d19cbaced31836d768adc7dff534810acf7c84a67cd0a87e4aafbb3ec959c04a",
    "sha256" + debug_suffix: "2fc24f10c926f7d220539d533c37fac39c2e530287a6e66f8e9c47cc5bc91798",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "c2ff54c3a0a3d6a1dfc703517afddfc2bdb7c28b7a77c03a86a1d7b58be5341d",
    "sha256" + debug_suffix: "59baa821129c09aa89dd36ff1e03ae97099fdf90529d6b98b16583b08f2aee6d",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ba3ffe3f4dab3aac1440260a7f9fae699a5e6a1420f1e193144e018aa5bd5da2",
    "sha256" + debug_suffix: "97ebfe5100d6ad2bf0bfbef881043b66c2f36bc540af66dcf3a868d3062a80a0",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "b6d94c726322ca937e0b962009f6839e3f3bc3d57ca649aa50d59bd15dc15778",
    "sha256" + debug_suffix: "a4d86c96ec895cb0909561150807b6d8263e43986ea1c6a9c267a25b067b80f1",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "50363226dac076ba41267cc35d2020f37355015370b218abde577810b6c0d8ef",
    "sha256" + debug_suffix: "60252ebf20479e8c69e73b937c5223943e7cc0fa3ced568d896da2eb68b90aae",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c0e9cc3d2609e76ba38c5195607c54ff25a40005235915a76f1594dde610e82a",
    "sha256" + debug_suffix: "352e220f4c43b25852c315166b3800d80289fa70ce542c98e1be236dfd3bf3f7",
  ],
  "kernels_optimized": [
    "sha256": "159afce8e362ab6721297b0107f94f3f16e819b8a68dc080299b22e1f3f2974d",
    "sha256" + debug_suffix: "3618f68dbafc4d5a63177997ec08639d1dd5e5433e4b7ea54387aa5ee34a1785",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b183000e0ab192d805323e15eaa3908ea10dbb422611f5ccfe65db8765531dac",
    "sha256" + debug_suffix: "6f59485634c878a33de6c772b7977945ae12ab25c95ee6c27f65d7dad0a86f11",
  ],
  "kernels_torchao": [
    "sha256": "2064273b7b76ca3f87c647c9c5a105ca3daf0f659c3745dcfedae217a79c891c",
    "sha256" + debug_suffix: "a70a717124f40b848bbfa5fb3dc25f0fee9a480a757682b04bd83e89e23778c5",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "96afd737142b8b5c95d3b9ab238ceff542fd2b53d7fb963ef0b23dd34070e70b",
    "sha256" + debug_suffix: "b82de835fb4ebd1746de07beb0f03260389e4641bf89da50ee363b8623ffc7e5",
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
