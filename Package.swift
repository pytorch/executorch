// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251021"
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
    "sha256": "f44a2633cb1630431a49e51c35e74928564f059923001f28e7183df20488f192",
    "sha256" + debug_suffix: "7854da429483514ca04e32843a585cf8520ebc5f6a71bea691bb167a54dfbb92",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "37d702bdbaf4b192ff75fb682097174fed95326e930fb0cc2c02baad493dbd23",
    "sha256" + debug_suffix: "5c822ebefdbc969b30349c49106b357d015f1c0b1d068a93f02bd8908a102195",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f5a9eeeda0b3b5825fb21768e42786ce598cb7ced8ff62ef26aa21eacae78033",
    "sha256" + debug_suffix: "2f650320c1ea26aeccd0d54aa8d66af19eddb6bd530b70816b82448001014ecb",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "432574ba747656953ec9a8fba5d282d7f3ba86230130238f768abe5952cf3db4",
    "sha256" + debug_suffix: "ba2d3716c9c362a2154af6d5219561945967c39f5b724d938cd393bc13fb1a5e",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "af5046de2fc95c8c76ee638926265f5a06fce7355b3283e39b20d37f39b942a5",
    "sha256" + debug_suffix: "956ec9938fa44d8a6427e96fbed810c7db307c0f88fba70f01a7f3b9297aa819",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "245936161aec81b4c0cd43d37048cea657c8bbd6383ad5149c11baf129898371",
    "sha256" + debug_suffix: "6256d5b99e145e6735e7b9bba812131d64dc7becbd8c2ab1c50d064d17a813dc",
  ],
  "kernels_optimized": [
    "sha256": "230da1a7ac8aa60ed853e93d5d5fa602166ce9fe33653f81e9950a967df9d75f",
    "sha256" + debug_suffix: "aa1daeaecb3969ab18611e11db8b4b5c8a2b4a67b2af4bf183b8647908936bad",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9ba14a344f225fde936979734d5245ec19691d6550fe3b82f5492f3cc6995051",
    "sha256" + debug_suffix: "c6a2bcd4bd0e95761f581efeff33f0daec7c3a65bfd091173e3686a364d4108b",
  ],
  "kernels_torchao": [
    "sha256": "9e5163ae3dd8f2b653d62c77053faf7dc3d3c813ccc1207fe5cb9b02f44fef21",
    "sha256" + debug_suffix: "2ab02c04d0059c035f9a11672e8b564250aa2a4bd46159d5092dc8aa11b61951",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "12487f15a1f376f5c696a7602ee0b927fae09d19ba198fca10eda9300bf1733e",
    "sha256" + debug_suffix: "85b1d963a5890579a12b90e4949dde2e7e52caa2b3f968023d431d9fc426b89e",
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
