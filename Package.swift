// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250812"
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
    "sha256": "31a71a8a2aebfb0ca31f2cbcbb130c517a725b94ee7d68c66464bac110b6a4b2",
    "sha256" + debug_suffix: "f23c8e2710417d6ce05ca60adbcebd3c0f0f2b15fd7dd5506d597ae1ede712aa",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b4412d6e71d1d7b0844861b555959cb2987aecab6469f5ff0ee50aa6edc89c7d",
    "sha256" + debug_suffix: "a25d271ab4def13a89f1484dc3fe76396f1f380ad6db14ed9288fefd8741b13d",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c80c2dab1de71bb095bf5af7c8b2b68b48681dbf67829944e6ed6783c8a12c59",
    "sha256" + debug_suffix: "48b3624b3393fdab8c3bc359f20920f658b3bdd9e15ab7ef80eabf9b62dfcf32",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "c99ec2aa9a98292e7d98a5e86fd8b3c884ca0fc65858da9d72c0d7856766698e",
    "sha256" + debug_suffix: "344df48528bf3b9e2b53f81b90e9bbc709b66ed338cb5c3aefd29e05fed27b09",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "62d620a05864f5b6b76d23ba7ca44635a6a19050770eb9f80e4e525337f000d9",
    "sha256" + debug_suffix: "dae6cb08d92f452ee0fe3fbe48829330a0a56ff18a8603287867f077725326bf",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "ec7e5952571f9cb9be94862b746076777db984dd36017ebb597ed93aa4b31982",
    "sha256" + debug_suffix: "14a7b7dfc4be094ba1c642a6112f58dcafddbd9db892f97a037268857490cf48",
  ],
  "kernels_optimized": [
    "sha256": "c61fe1fc8fc76cd3bda7cc7d058c9212164f7439295a7cd693f71add2268358c",
    "sha256" + debug_suffix: "7c673fe565955a104cfc59b6677c3bf444d469685728fc2f66d142c795fd969b",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "f27b2f8567a3a7aa5a4f63e2e7f0dedd0eac15ae5a2f43004af74feb12d38aee",
    "sha256" + debug_suffix: "452b5daeea16c8cd0720d32f129dbbcf04f25a404d37ee0f17c3528b7795d401",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "bceffcfbad56712ed4f7d8930c46fae25bc17493e59f4b002fd61b3186cd3d35",
    "sha256" + debug_suffix: "1ddc9b1c360009a636a3e0b8359a1c61f54347029843a40e71f9158659c5b5d8",
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
