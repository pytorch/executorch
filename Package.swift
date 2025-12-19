// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251219"
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
    "sha256": "0b32932c57874093cc4d9c573c242b8a46c8afb2b026b77d15fc43ce8b54e118",
    "sha256" + debug_suffix: "952151fd7e20ff82434722e28f1a4e6ab32b35f77262d80950dec964f1b056bd",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "4a632c9204e48df3e999e9598318c5d741b206ade0e9282ea228e72a2f42dba8",
    "sha256" + debug_suffix: "de99274a510e467b07abd077c2218014e214cf36f11621a7305cb35eab41fd56",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "9d71b89ca13c506969adec67ec5d095ca7f6000a877892a911e99c55517d7637",
    "sha256" + debug_suffix: "5206f2eda413527528086a24ee968336aecbae9b7ab7932eb9c1cee445f7130e",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "3ad8f748d0a49cbf04cffb10a6ed3f9f191045f2ef33fad28bd2a5f9eb255f78",
    "sha256" + debug_suffix: "c48e81a474cd74d471f8974ccf52da132c3dc19f8d2d25d551ae048b800513da",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "d1cd5261c7036c4eb8ac3441d3bfcc2137afb9483c338a6fb92c361a478c8a1b",
    "sha256" + debug_suffix: "83b5f1d6403dfec201b5a21fe6bcedfff16923fd561a9d66b2d10c008c703578",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "72dbb35baee8ae57f94cab102a114f18735604c354de7f5e7bf127d03868e198",
    "sha256" + debug_suffix: "750b08bb10573d5b00dfa3874d1a65daac48c8b5087a1ca36e5ac0b69b2ff118",
  ],
  "kernels_optimized": [
    "sha256": "febfdc0fead487c1828ddebb6254bf68ba51d80b26df7b57bffb39f1256abde6",
    "sha256" + debug_suffix: "c052f592a48b9544acdc17f16b122c92a91f791a765ce90ddda36965359e6009",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "8cf523bb1e18a4188c7fb7f3ffeb7f6e0ac6b9488f28db7d5caf2502106930c0",
    "sha256" + debug_suffix: "66a026f612d3a217de38518549e5a84d7651502087ee08356c2bfb2f91587c56",
  ],
  "kernels_torchao": [
    "sha256": "ea86e3855f14a55ac2b16601bec67444d2de51e9c855f4c620181e264c39f74d",
    "sha256" + debug_suffix: "7bb2aaecf08b19bbe560048dded305c6bfa4a7d41032f19ee11008ddf3cb3eca",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "8227d014ef14a921a16e13f375b3eab1cf8585269018b5ff5abf39eb19ff4c8c",
    "sha256" + debug_suffix: "851532313d9531e0e49d5a4af9256d9f40130468bc12db8d0f023b15eb903da2",
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
