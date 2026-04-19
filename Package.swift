// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260419"
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
    "sha256": "20df9d106465986ea4e0afeafbf674e440478a4ff4f9a7e135cfa9f3b4b71094",
    "sha256" + debug_suffix: "159c24a49b4238f9ffa733a5ebccc402463ff9190227e053f23fd42337d790a9",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "9510b5a9848a783b4e2b0525aa4bf7ae9cdd593c092054dab051bf690cf47638",
    "sha256" + debug_suffix: "b4ee1457ac97f3f9e21d436808e444f7d23ac0489f1d2ea1cfe1ac8d1499e63e",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b34656002f9235d1e88209bc7ff7bffc0e2d51c28631197484bfbdc03802d193",
    "sha256" + debug_suffix: "555091e8111e4614de89668839a988d390babf8018ca43548671a227cb0340b9",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "71fbf7d6bb4a45f8da8b5985199213178e657b93893f57e2a99ac31342e4b6a5",
    "sha256" + debug_suffix: "f93d92c8a82ebb2e9007184bfde40156ea2a11a03aa4cc4f306d34b75a5bf75c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "cadf17ecbb39a66c365eb1c0d7c33082e583dd2e7c43c67e2af4ede57482ef77",
    "sha256" + debug_suffix: "0e63e14685eca2c074d3a656c88293771b13c4bf78592ca9e35e9de8db9c621b",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "ec18dabd73621dc5891d6f2f0f2447157231699f5b236a6836a804a418862bca",
    "sha256" + debug_suffix: "4adfb1d59509fbf6bf02c132d1e56c6ba386aa4c4404018e3a001c02af9bb827",
  ],
  "kernels_optimized": [
    "sha256": "c01fce4709e98568f2bd5077bc520ef45010e03e34a66fe5b793f78c8eb5e2f7",
    "sha256" + debug_suffix: "8294602bebce0196b3dfc7353ce1bb60f1fd665be8aa10ba691cd73e99274e59",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "4be2fb25eb88328bc269295a0379878636817dc22d9d60ac78bb73c5a38891c8",
    "sha256" + debug_suffix: "874216f328ff7143bc6b94573b50b96844146af049de2a576568eb55b5110b8c",
  ],
  "kernels_torchao": [
    "sha256": "cbfdf54a3a7c7d5ae8e187a50b79a7fbe62ecb56cad9890d4ae3c46de6a32fa7",
    "sha256" + debug_suffix: "04f6573cd3394ed22a5d6b3ff6146245efd82436da1f660c8dd47bdae333defc",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "7644a20c9af252d10980dba52add78a85569730c3398e9c285ffe12d5cbbd215",
    "sha256" + debug_suffix: "fe5f759ce8eebf9bd07675378dbf2c3ef4f9bbfdbe50f7e5d8e585dbc57828ad",
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
