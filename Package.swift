// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250607"
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
    "sha256": "54844d4611d881271eb71d252133ed887a62c38c6e96adb399c7f6ea0df41a8d",
    "sha256" + debug_suffix: "4cd44ba03834d3f14c00b4ff3344c6c10e7f877cdfd0f5c681f502deb212e93a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "32b93b7f3362f61d9baf5e9e1b4ce4385f74930c94347140dd24452734c69ec3",
    "sha256" + debug_suffix: "17d43e8040971849c90ba44a02ef5c16da29121c9d23f7979a0b51e025434fd4",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a2cc49cac2c859e838e3feff98ebdf9950d9a80407e0298f093b58ead2f9f3c4",
    "sha256" + debug_suffix: "96bde3978db5f02fa5889a3530b90397e0f8d4689e6cfc592f25492091abe94d",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "48d90709334f1b1b6eb4dc556e4b500f4ad0566db771e745808725b9d6fc1292",
    "sha256" + debug_suffix: "94d2a3a904855208f1f13e621846161106808258126ffdc46f01fcea4b1a1f9b",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "44e4d9742b011ce6df6b531141fac3db5262db44109d096d9fbaf1628728450e",
    "sha256" + debug_suffix: "2114f4da214e17dbd5cf1968f5cf98c19dce0cfcfce6e828d37363bc45e05647",
  ],
  "kernels_optimized": [
    "sha256": "e34910b4f829c1a39231ed53a1b1a49288d9707fb146a5e0371e12e3b03f841d",
    "sha256" + debug_suffix: "e02bc3e9e0f0d8181ef47731a2f911bdb4f98861453371111ba7701f800e6d10",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "cb34df08102ca626e712246b3a999adb9b452c61c62316e63ee61dc97fb912c9",
    "sha256" + debug_suffix: "653f64d6bb49371dacaa3ad767cdb7f17dfd0ea58adeb2d3759a98a46ba1a2f0",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "11fb3aa73f7c17aba5711f4508d380c6f4a7f078bbd8802bc5d155abd12ec608",
    "sha256" + debug_suffix: "7dceb5f6ca98bf12872e9310b676dd164b1143a32996061f7a33d41e4e41e05f",
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
