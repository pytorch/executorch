// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251208"
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
    "sha256": "d11846354ebd805d6cfff25231e8179a3a71c2e5670f1f12e002ef898fdf7455",
    "sha256" + debug_suffix: "3e34760d6761c1967e7957f8193c6a481bc32666c72b4442e274e718745b5d1b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1ba56d7bee9644674de60f35a7904a7419048e269e455b64b5e07d480f3e18b7",
    "sha256" + debug_suffix: "3cadebef69b8afc10bf196c0f64f1ec9e2632c4fae54513760a02de9cf48b333",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6b792fb60496599152293ceec8bacba991763d1bd33fc7c512a0918a0a5535c0",
    "sha256" + debug_suffix: "cfa090e24a020cbbaed970b84b5e955d27acac280468543833335e807d38411e",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e168e501921dd3b212f28a421b0702d1a752498c57de2abdce4daf40143501a0",
    "sha256" + debug_suffix: "e37eee6f2896ebf26d174da3364202ef469e7b693d388c3e2169396bcdff4941",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "77435e0902702df80a737a6e6a79cba3aaea899235a397c592c638944c8ee13a",
    "sha256" + debug_suffix: "bf32ece89d358d78dd4022da64b7e380dab0a7bd63e75fccb2111d11e8237c32",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "211b3e4d25e9edadb1562fe38d9c9e70bfb6706b8a467fc6b1b9a9fc6e38b69b",
    "sha256" + debug_suffix: "aa9d2242fa1ee5f7dfcdcafd95a3e4e7d71421057effa85aee9e0f7d566b9da3",
  ],
  "kernels_optimized": [
    "sha256": "5a643b8c53667c8bb6a6661065af7223f5f8cb07f2f0a6dd424e71565102bfb4",
    "sha256" + debug_suffix: "ca2b88f5a02e9a81643db4117cd223362dd908fbb23a92cdd07a2e922b90d245",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "c8604bb65433737437612e30f9554d8f787eed8ad024dc23cb2881220eba2c7e",
    "sha256" + debug_suffix: "c8a7b0774f8afe1893ed71287b9f93f5b88e394cb050f669a7c3cd9b80e892db",
  ],
  "kernels_torchao": [
    "sha256": "7404d9c159074326412b8cd0769d626baa742cfa7378e2059bb5195ced394416",
    "sha256" + debug_suffix: "d3bb9c22de57bd3e70696f4d1bee006cbc00dc532511582539fe5ead8a00eeca",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "ff21b81d337497540acabaf94ec73a1bba1bee05178715ed42f7210f3c9f9770",
    "sha256" + debug_suffix: "f78f2bfc63487ed7c13734130ea77d1529a3a14e99ab60f42b82571614283a63",
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
