// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250903"
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
    "sha256": "ddfafff0db05d5a20f21f4b33e1dce9a66066278e89d565495c5d682830fb052",
    "sha256" + debug_suffix: "7446c1c72a78a3fdcf01602d073f7bddfcaf5bf0abda9ffeca3015a75085145e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b7d566ab2fc40c7d3d9b413ee39a457d716da8952960126a1fc3bf1c859d6a02",
    "sha256" + debug_suffix: "5b92d58ebe48790260a89ae701da5ad5da6b24d82a2cc9d3fcba5abe290cc4cf",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "cb40c67dd2333a3120805995615edb601caedbd1e67835131b00617365f56286",
    "sha256" + debug_suffix: "a980a83f003008e6e57edd0e9b621c908171d4746945b61168485837b9807ba3",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "a72fe42fc24278fd893e8bb9a547c87b4a4c2618df0c3f83a8c7c3a07b2d93b7",
    "sha256" + debug_suffix: "36f5de33dde9073d09d3b2c2aed11f3a3bdb529594b80410ae11ce4acd129358",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "9e4fda6dc18c51e5ad43a2c19725e61ba3b83414ab14c3c932ae53ffaa3d1f09",
    "sha256" + debug_suffix: "8c9476089f32360ea71a7d929d144749ab263d3380d2379d624eb7d33e9cf169",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "250801a83b403c00d9fddad6e9590ddad97cd61728c1fbdb62e925b59e5799b8",
    "sha256" + debug_suffix: "a8bed69aaf6f59364866a8b9750cf9c778f01c066a6d9dbe2e5bfd9e69612130",
  ],
  "kernels_optimized": [
    "sha256": "e0a5792471a376f99185cb5a883647ca20eeff65ef63a5ab27aec40591a268d7",
    "sha256" + debug_suffix: "83491547e5a8ce0431c4c50400784074e59fb79173da7c608e860e9d1a03c254",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "daeacf4b8093bb95327ef5fad03f8fb57b0a94abebcb7d7b2ee645c53f6aefa5",
    "sha256" + debug_suffix: "3098fdd59b3cd84cc2074627879470d42212f7b6ad700ef038db5af86119dc11",
  ],
  "kernels_torchao": [
    "sha256": "3c07c827be1f44641d4e28f083709e90e59d43d76078f825d751c060e79cdcf6",
    "sha256" + debug_suffix: "8e07c5b3212e1ef41ea90c47d62e23ae18090a1ac26fae044cf642a6e59ec023",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "3d49930becce0444670d971463f80173270d57278bf31bb69479487e6b250983",
    "sha256" + debug_suffix: "0b7690d7d4068a7d4f6339650d1382f4b37cf9165b9efedb957eeec497eaeb72",
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
