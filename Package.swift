// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260503"
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
    "sha256": "dceff9ce89947a2500d3adaf7cc2f96f871eb0b2edc93c3aaba330e19f78dcb2",
    "sha256" + debug_suffix: "56f65c32090a0ea99b7504eb655e008049b0edfa336a4733ed60d814149e07b2",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8dcdb68a6a4dd8a3f560edeb686d10b81b0b7f19f07e3a1804a8546bea7ad6c6",
    "sha256" + debug_suffix: "1cd5dbca7efc944d34a38b15dc4c188adfc5c3568beb428a2f6f682974d56a23",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "9db165fa2ffa2363acecdc0e243e6446e38ad63b357351a4e772f2481e8a2397",
    "sha256" + debug_suffix: "705f128904edd4e5607944444c707b49c53c4a68e15c5487ad16bf2e56157558",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "cda471f1a7abc5300f55ef29225622ee091884ad77c3b0c006ca38ad2d791b68",
    "sha256" + debug_suffix: "be868d6c04ea259320b948729349b1219d930ea25edd32524275a40f316e368a",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "74ba3e466b6555e0ffabac22e7cd0d95ee94150550fe6a44cecdafb6f31c4be6",
    "sha256" + debug_suffix: "09f20823352b853085e2d4d149cde80d6160172b3c17fc37c837803168b8ab0a",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "7acdd083b1993319511c790b58e7dab6f6c6729098c467e55c9ecbeae6966107",
    "sha256" + debug_suffix: "8ff954e18d916103092984954b98a12032870fdbbe2be2ed1f9d54318c23629d",
  ],
  "kernels_optimized": [
    "sha256": "a1d187135cddfd130ad1439bff26570bdacf95fb61c703adbacb8cd3dbc5b739",
    "sha256" + debug_suffix: "b6b684f986f248aae6c68de198a9f6f6534d3a2fab400786a49ee94022c5ddfb",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "27ecfb89d563c2b67aec0a5036491af37aaec18f9a3ab558b5fac429dc21836f",
    "sha256" + debug_suffix: "31d1ddf3918bb9f742936a090a3e57cbe9203b7d8a861bfed9dc5e20f49fc9bf",
  ],
  "kernels_torchao": [
    "sha256": "3e6527cc5d271fffa6ece9c1e94e16f66dbdf30e9dfc08c504bbb962f2a2cdc6",
    "sha256" + debug_suffix: "13aa34bd2e9bb100e3e5762cd14b1ec989261937d32518f73297dd07ded55dbe",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "c7159101c2dfd0d516b150cba93bd688eacb21103156ee7158f47b99dd080725",
    "sha256" + debug_suffix: "4508b05ed6550e590ea7578eb616225df5b29cf4cc349c189e78e675e8e6b8ba",
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
