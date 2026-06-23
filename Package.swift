// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260623"
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
    "sha256": "d6a051032c0a9a96cce116147fe31f991b4ad4f431f03e18d9aaddd843769f24",
    "sha256" + debug_suffix: "68122f98043ef589b02ec15a1003152c2012e2513bc486aa83fca5f16638adaa",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8ba851e4168a35a6a46a19fe7c63876ed9489530a2707f9ae15ec3b894b6a0ba",
    "sha256" + debug_suffix: "bc8e79045f7b8a43970b9c61320d3f2a4f83541d3e5395dcba4c020679903070",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "06a8788c69b0f96400e8ceba697308a81ea4c7bf2f02cf23ccb9933355053148",
    "sha256" + debug_suffix: "ed5e1f21cd49e575a59868d220628ce78f9dd25e4996a3ffe82ae185e9c175c5",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "c4ee272cfa269a7e3ad9b781a081747d7eac7ad7d7a326f6fd5590a0bb6c02ea",
    "sha256" + debug_suffix: "34b0ae662b54e2c432fe63a35f69fc4ba876686a6071080ceacd3287cf771dd5",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "7a83cd7550f59c9ce9760d46ef43bec5d07632881aebb7e08a939a1658639cc4",
    "sha256" + debug_suffix: "7cc66983fff73151fc6f290020ef58d0c426edd7537df4062ea724bc474db65e",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "37eb2d574589e233473ab602aada094d004ae304e1971bd830708b2abaa2b2bf",
    "sha256" + debug_suffix: "8dc257b3b4db63740032eeee2b89d406bc1140cbbe4195d39d6987da32151a55",
  ],
  "kernels_optimized": [
    "sha256": "0f36f1665d6c5ec8e93429a47782d81c37ea314606bba8e6ba0822ac43405c60",
    "sha256" + debug_suffix: "0913ec42bd866a791a3d3381e8a89a441e53a12fe02c9ab9efa158f804f143b9",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "45ee30c4e252b66540b93eae773dbeaaedaaadaaa398af0fafc6f31f12f3c65b",
    "sha256" + debug_suffix: "3cc632dcb5204fe0049fa7769d2999ef59cc77aa5dce385507f421bd7f30f18a",
  ],
  "kernels_torchao": [
    "sha256": "db15c1127c88a184a238861068358c427150ffeb8122fe33f7cd2b35debefa49",
    "sha256" + debug_suffix: "a2331b844dc7f4aea62fa22881903065f403c53a00766a4c64fc9df779f984f8",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "c850efa2a187951ed7ecc9f3d42961cd3fd4722daeaadb4169cb83e2e8f025f0",
    "sha256" + debug_suffix: "c2fc2afe3692fb2ea517d6d8ca03ede311f2aac1095fb4a2ea8f264f4202e046",
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
