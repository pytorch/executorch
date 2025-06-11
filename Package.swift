// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250611"
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
    "sha256": "81b22db7f63ee47b7c73293abdfcd75581c6b8f0e5fb7552eeca7eceedf6ad03",
    "sha256" + debug_suffix: "0c79286b38fe73be315db789c9b7f6a9a92f8a2126147d7de4699415307d3514",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "da2821952533120f0caebf06a1557f7e7c85ab7a46f185e6a4a75f3259b82702",
    "sha256" + debug_suffix: "fb4a8db5623e83d7d1ed2e6ab46a618271e9577aaced5b788d414cb2b98e6b1d",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "372afbf36dbd1640146d9b7d84d4e07944f8ee75e5b00ab423f403f6bad75ef9",
    "sha256" + debug_suffix: "b83363f4e76aa4ead3782374798e12c8e649b9e418b379410abadcfb846db89d",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d9a2ca5248a518607fc8626a77832347502576a1696a68be107eed93fac16783",
    "sha256" + debug_suffix: "c12c98eb7d809de3f72dbde1cf3b27828f374bd3e44af7881307fe4d046e7d54",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "4c2e52ba185baf18c37d77daa67dfe2145af13ec8094b897878c9aea6edec034",
    "sha256" + debug_suffix: "5b66c9750bde71c26b00444d599a638be472ef31114f2ec3aa5934e2f28ec54f",
  ],
  "kernels_optimized": [
    "sha256": "fe50dbd8ad357b9e171a7c5bf212340408a68020eb2e6c951e3e02ea4039dc71",
    "sha256" + debug_suffix: "b00948ec62728699d12cfec3a8819102bcab143cd02013cc07958a635adda483",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "cdebc02f1697d341431d02be4caec0ed8fcecb374fa4b83a3bc7f8d652abb4ea",
    "sha256" + debug_suffix: "1d6a5848e4c445895829df57a4dc4eb3a4ef0ce951498d2ac2ec31776f014d9b",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "3ba5001c97cbe78073d2358c2fd290e22e151e99ff30b7d4d8d5f757dc6c22fe",
    "sha256" + debug_suffix: "9a6e93c885bdbc5d229000cea2b296d9150df0639963bb12429203f51fbbf863",
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
