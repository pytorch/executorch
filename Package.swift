// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260803"
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
    "sha256": "92a405db2f369f74bbea62d528aed3a888d3512d000ba9e9b6c43c47e907a5b7",
    "sha256" + debug_suffix: "9bd08e7986ae3be6ea27662fe8a27e54fcf27d10569a19bdbb2aaea60041591a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "e8720ef9dabcf38b7c5dcf3aac7f0c298cb8ff9dc5127a711ba2f3c5871021a3",
    "sha256" + debug_suffix: "20670520809fbd5a423f2152c5431e2051eb0e9f212a6516348ecf91cc4fd12a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e70fad96aaab74b1c281e58fc797583e92a1545de73ba6360c10c568f823c42d",
    "sha256" + debug_suffix: "c09a9eaeeb8a5a658865013de5ed14611f1a3bb93649d503f0e5b31eed9908b2",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "eb3c260b5e7e556ff2c2c0fc7d121efad94327bf46d7d67ad8f39c22e6cdc55b",
    "sha256" + debug_suffix: "d4afa9fac0e1012be71f8cfa1cfa98ffd0586bd847e7d80725632f94a18ec608",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "679c31d3fe0d4b4ca958e4f68fc72b083b82d0c9d1f83f706d72d1b50e537825",
    "sha256" + debug_suffix: "db34c6d32fc32f3e77115ccebadac445dc94739a21233d5a1db3c450aeaa24ee",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "e4c3b0407ad9f407c1543f89f21629432a0a6e7fcc3d1ab2b235a25eb3129dea",
    "sha256" + debug_suffix: "e94a3552a8b72c4bc3a5622c22e2df727cafe2159052a722f340f01fd822a5f2",
  ],
  "kernels_optimized": [
    "sha256": "15bba11fce0a3f82b92209f2e1e3907a05f0d698f3b18881c733197a3c9431a7",
    "sha256" + debug_suffix: "2e4b3386084d8eb054220d7e0f02d41845a228ede045121046fd0897bb92c87c",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "abc4ff7496b51df2a05ea40a3ac24f32e8f6a25570e487cf68fa36121fc73450",
    "sha256" + debug_suffix: "f24450bc42324401ff6c7a5485cd6f4a4d3a46b33d7782b9939682b0cd810b9c",
  ],
  "kernels_torchao": [
    "sha256": "c87e02452221d70861835e6373599a70cac9401ef043114b5798b5047bbeba69",
    "sha256" + debug_suffix: "2db3a288d7897819a0e72041f358dfdbde9b219bf59cccc15e885de96fa0b4d1",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "7d0c47be93d22f0a47d58462fe0dd3405c22c329f2ea39f85d0ec5fec2c63b8e",
    "sha256" + debug_suffix: "b73ddd406962e9497385f453468d4ccdd4e1aeb9a71eaaabac93654ce1816e75",
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
