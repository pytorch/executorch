// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260513"
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
    "sha256": "d8bb79548431fbf8b01b0ea81ee4c6365e6de0366f78869ca1707cda19eac658",
    "sha256" + debug_suffix: "f2374ae8311cd9cd82db2876acc747a356d48257277256eecb091fb9ed8c176f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "9201b1231da82cbdd1802d70c433b4bbc8ae14a802c79ce86309bf4686f77bd1",
    "sha256" + debug_suffix: "6cb158926ca06513c6030bbafd52ab1c562ae94b181180afc899c26c128cc17b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7669b3425c81ae31daee544810d6a13e08bcdfe8b9d6ed69418e3f68a974dc2e",
    "sha256" + debug_suffix: "2de2fe1b0eadc781952d4f7c8276474dabac6f06e4b7c72e0360d5442895c0c7",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "fe4cc961d10a7e8afad090a7a3abdacab2aab9b60c025adda36ba729db0ef170",
    "sha256" + debug_suffix: "5de143014734d3539b1dad319baa1fdb7c5f889de75946e7ccb7aae8a9574f27",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "a91f6db4fc05a11635343192e4c1d6c724b683bc4ce837be6bcd362e4999a969",
    "sha256" + debug_suffix: "5e538d8a76e949c847ad0173fe22aac2deb11c9d9f434ce55ddf1f7984a60041",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "65c191c2325c5ebe55b8af22d9d4f004e4b6e1dc4c71405face89fe8bbe5ba61",
    "sha256" + debug_suffix: "c5dced1ecf41ed22dee544e9e7f2dc9b4f9b74d4374dbcb3a425910dff6053f4",
  ],
  "kernels_optimized": [
    "sha256": "8e57bb87288bbbf83fbe8260f64ea890a6a932290cb9c563fc4b72d898b12f78",
    "sha256" + debug_suffix: "04477bbbe921375924bd7f7c2a38df6f69d2901752d9af0f004deacfab880187",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "795e0ea79dc15b3f6d1aba9035c408b9f422a136f0db7bc54bfd789eab9506d4",
    "sha256" + debug_suffix: "77e009fa23c974814929d314ddb313f5723623b2abd07474d6a7bf71c25e0251",
  ],
  "kernels_torchao": [
    "sha256": "153cd25c1aad3f007daf96087111b61902d0fb3b96a060ad7aff5b044740c528",
    "sha256" + debug_suffix: "a725f62d571e501fe33bdced226803e9a71c40c3fd86c9aacb9914b4d714e36e",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "27f4fa650fe9eb7a74e323b88b965b03fd26911ffcb83d58936d6540b3c2c2c8",
    "sha256" + debug_suffix: "5ff1f489519d241b7f005f3b17ff558128dacf6b5f1ad48f9216e9af163812cd",
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
