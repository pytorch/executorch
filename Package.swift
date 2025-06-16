// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250616"
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
    "sha256": "9d7fb2b7194f976ca2976e31e8404595eaf2472d5a14915595dbc5af385231cb",
    "sha256" + debug_suffix: "5066f3e3aee175425f1255c1fce2e04e0f15910a60e4751e137510c8b02f704c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d73c3610abac23f83bec2feb8e664619d4b7cada1463a8d746847aea045d6b5f",
    "sha256" + debug_suffix: "79db342c7fe968217a6a77e2b0359ccbc02a5d51e0f64f127c33cbc083ca5b68",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "12ed102d72cb3c6330f8d56ed9c8bd7b1de26bf85342be7a87e31097b0f289be",
    "sha256" + debug_suffix: "0f33af9a9a4e57727144ef889c403a84cf5af6d12742aa593a5c45c8e0c31891",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f714a33eb352e893a457389d8da7873bf0e05bc19e0c2cf8848688a6c0285dd9",
    "sha256" + debug_suffix: "c9f92a9d08b0f6430578fa19b1affdf27b8d3366f50627374cf663139bacdce1",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "849cd3705eabbbf5b61b37a77d4beb7a3b460d97ce4d1a8ae4e2209cabb17a79",
    "sha256" + debug_suffix: "991de9b8620f552e58dc1237e7f97fc261e6c8adfef3578e005642eee5490587",
  ],
  "kernels_optimized": [
    "sha256": "ac8825fa06d893e258c289fc46e23d158b08a74d3532583faec0336abd1249d1",
    "sha256" + debug_suffix: "385f4168f7f2c2f366727c0fa569ad1e1620529ce6e90c560fb5be591420d1eb",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "4ed93998bf43cfe4a4ab66b6660221bc9ae8ddcbf69502871abbcb1c5494ad18",
    "sha256" + debug_suffix: "e9cdcd22dec90b53cc51539b42f05d2d89782cb924a65b84acd51310579ea035",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "24b6a08655ff5fa1b4abf89bc5f84f3242177a60585c9c036bf03e05e54a0c83",
    "sha256" + debug_suffix: "b0faf807a0fc2b3de3da75be7bb548a9ce1d4d2bb7ab1a568554009dc537d7b3",
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
