// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260130"
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
    "sha256": "558d81bc9f3194bf35ce459a3d4a4a394997e301b013e3be95c8624bcb97f4d4",
    "sha256" + debug_suffix: "8fe950d34749379e59c7a40a185df15249d2f21c292d2bcae8f80bf31434bf72",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b6c9afe23f85eebdff9d6051eda82875e1d03c98e4f41bcfdc984b2aaa9af213",
    "sha256" + debug_suffix: "1db77a6550dfff7ed3e326e82f4c67fa880f15307d823746d54a8d3275d897fa",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "45d749221bc4deacadff5d9561e813b1379dce4645265405342b8ffe00c2ab66",
    "sha256" + debug_suffix: "afa2d495122da22bb93d5bb7913ad16ae0372f39792e36b8f2f28ae8aa5c6fb4",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "94879ff61bcbb591fa1567ebf1251a2f6a4e5c273cede9b4f8d316c6a3e77f15",
    "sha256" + debug_suffix: "017e697babd255f86713f7571c2a70fa0a8325630efa9c3fbee445b3f7e24369",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "550c169fa9c657607619922564895a4a01442b79cf0f6925b55a23a5d46ddab7",
    "sha256" + debug_suffix: "0e0f203c9ff3fa3a811035d4af5a080384ff4a82738ea76a26f2d3f0e32fb647",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "1700376fecf6b9b9f7c9ff9da13d6b86ddc7ee18a612fd59280f8e8a945850fd",
    "sha256" + debug_suffix: "3279b8a346f7ec9f4de8127f4e01d016aec4bbfed3c20ec69e14709dfae262c5",
  ],
  "kernels_optimized": [
    "sha256": "669f708039a11e98aa8a9cdf40e8e44775f4f3c27b1eb5f838eb72661534b98c",
    "sha256" + debug_suffix: "fb3525094c540fd191ab1d7ede4a1ed3c68839106660d2074c4c15566557e9ce",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "99859b719bb89f5b32895e6afe209abe78bc65c1679fd795c8fd0412318e3d40",
    "sha256" + debug_suffix: "f6b511d87ee11ed3505355b423d3f168da256538c2e82061f986e4a79ed1b759",
  ],
  "kernels_torchao": [
    "sha256": "f6138bff8e35029c7cbaa8d4a3f3afb796f27544e9df796c34c5b1769fb157b1",
    "sha256" + debug_suffix: "a474714f656f23706898a5fe0bc44f8da697654449d349604acc748ca60b8fad",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "3e6519001990fbd128b06eaaa507ada8b589fe42d6d211f480dd5cfd383f4cf6",
    "sha256" + debug_suffix: "376a99a63f73ffe539da1b3bb920ff135a680c243d29518c4e4d1a9de01f06c8",
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
