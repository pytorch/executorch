// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260311"
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
    "sha256": "fe84607a7bbd329692549a93620610f48afa7bf194443c4e8c8a20ab39f1846e",
    "sha256" + debug_suffix: "2a88f347f969163a19cafc9e4b0f35c588739410368b2f37f5ba22eec9ccdd22",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "314341c1d927ec4245c489da3612de2446316dfeefb2331692a79e7d5952d369",
    "sha256" + debug_suffix: "0bdc08c92a9dd8491145205ffe6ba3c272b0c41e0d413a9cba7cec0e2e2e9a52",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "746e90dee4f9517824ce521fd5759808c1c9e3e8b96ee6f05e8c187e5a2a1afa",
    "sha256" + debug_suffix: "31c3764ebf1988ea805cd30fd2de7b3ea78e2d3ccc2f4c46c7bfe62e350cc395",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "57bddd1201e071fa9996e68104e5dafa68c76a6dbdc9e5f139882ebaa0728ac6",
    "sha256" + debug_suffix: "b6ed47c7d05d7f4ef226ed4b9ecd7fa8fafaf12006193e24526d8328707dd9c9",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "89d9d8b488201f1afdcffcabde00dfc47344235b98c26c2d6ad32f3d6a7ec190",
    "sha256" + debug_suffix: "4f1ebe3d970ef81eee14df8b7d2608196b34d8fa6e43959a29e385a937e468b9",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "8ac26290e3ec5dd8fb3717c1d2ea0526802daa59cbbcb7402bc939d82c63ee9d",
    "sha256" + debug_suffix: "096286ff5778074a7d6ca7fe086d766436d7724a2e64d21ad6971cdcb8725f7d",
  ],
  "kernels_optimized": [
    "sha256": "536b74ee67dc8b275f5b5c2b96fdca2ed1adb8608e247437d9a72e12d7eaee29",
    "sha256" + debug_suffix: "5edfd5cb5f450c6db0feff79c139a582054ff702eafb78daffb578fee58dc2ad",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "1293c5a99bf335a9610ff3b68fdf2e8b2d4bee9a6a67e27893bf82f438e1c019",
    "sha256" + debug_suffix: "d9a5ef2d36cc1bc2e90f0aa6e070ffdc9ca8d17c9c5fe46247f4ae282c780925",
  ],
  "kernels_torchao": [
    "sha256": "757505ab94b810a1764de90e35a6d4de10fdd9f7c0b08b37f0160e926180d32f",
    "sha256" + debug_suffix: "f59fbfd3162f63e64237cb4c96fe3a44f2226a3f61d7f1d77de0f74118dc3acb",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "95f49aaa215a0ce23cf1d34f1eb67103b7d4790d4fe37e9c3d27820dae133e57",
    "sha256" + debug_suffix: "90e4b5e460a0cb86a321e42c4630ff65cbfc4dbf076c47029df567a17dae448c",
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
