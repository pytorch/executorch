// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260524"
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
    "sha256": "ca75f64ed0315aa35ed12810cd228dc653b44f17b623ee2a397fbe65e7c0a99d",
    "sha256" + debug_suffix: "caaa0b91abedb4a95e7544894410b1f6166b31b1edf099445ec0dc8e36781374",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1a15afe3c0c0857792c9c46f3305cba8397949426796394bd0b25bea11c94552",
    "sha256" + debug_suffix: "39a6a119efae6fcc3060b0fb24bf3ca4268a745701bdae2e6a7c2676d8c5d991",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "947d32bd234468dc21a0d1cf13b9704640d9e802494d591ad8c3e42c6e121b94",
    "sha256" + debug_suffix: "e3a045241dc43fb8a2e74d1c4682da01f14a06ec9d6a51c79c3bfe87093a2bdd",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "4b02b2c217add10ac379ede1638686e3d3087be83d0b2dff7faf601d9b1fb94b",
    "sha256" + debug_suffix: "2df4a8ef278580a1825a8c0e07196179947ca6b8bd1a2167a77468c1b07ab01d",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "4cf25eb0f89f4ba8a21aae23a06ea777927583445d15371018e33fb870f251ab",
    "sha256" + debug_suffix: "d1d96791531f3e6ec406d2d7e96be5ec561bce5ce186d2648b3a0cd821729269",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "71a1ed1f3c4943749059cc286b126cd532de8670f7fe373664658b8f064cdf60",
    "sha256" + debug_suffix: "d9e7718ad2017dfde99170345b86f284e6f7be7cde67963565a41a31f3df735d",
  ],
  "kernels_optimized": [
    "sha256": "a66204983aa7f55a7268aec3deaf186a75ebaf0f55b7cdc09cf8dd5391ec3dd6",
    "sha256" + debug_suffix: "a68856176036f3d3fca1ad7f85492ff65f22c92722b6debef0709c3e3f74fce8",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "075804917acf3dd3d93104d560b8aeafe814c08b3739bd2efcfc38ad10cb8def",
    "sha256" + debug_suffix: "a60555283a69c0e7033d9738323006c205eebf97d48c53c44bf9d294fd29a0c5",
  ],
  "kernels_torchao": [
    "sha256": "8259b5f8ab9326a48b79664a229b2accbed18b47735273554471fc27a0f3c737",
    "sha256" + debug_suffix: "71cd8659f4fdf429d7a4186834958a26379e63b6743d13938d073a8352a0ca3d",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "9b16b59fa3dda3f63a061b2d34b52d3ec6d3c899224ed10d8204a7c122707db9",
    "sha256" + debug_suffix: "efda5c7b7a27e153e0fb7e207ef5d8f1a78dbd6df80dd0f537f2b5ac9a271352",
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
