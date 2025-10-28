// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251028"
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
    "sha256": "fdcf4cd16b5281446dd07ccdce973ce631af15e64796f5af42c0fa2319448fbc",
    "sha256" + debug_suffix: "0eb8a878db1b379458c5e15700a7f9aace038aa198c4cbb56b5bcc11d8e0e335",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "e8c162483e4484f5b7c61d18390d1990ba2e48ae4378d035ba1ebcf7f6556086",
    "sha256" + debug_suffix: "9fe33f219453415820494fa5a9d502a0eaabaa826e2e3679f790a09fce143bcc",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "951c51d1ac951810881ea1d92546b4ef1a273c1604a071550729ea33899ce698",
    "sha256" + debug_suffix: "908246980b836e45af32da12e4dc6496b97c02f73453cb4591e3b5158babeb20",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d86eecaa8fcbe553e2296e5e315c3f68967c775e68116c4415c0215c8f7923db",
    "sha256" + debug_suffix: "4bbb42763f6f84fb8723dc8d3fbfae129d06be3a8dcbe728b65fd1a2c5305fb6",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "87bf45c82b11db01ba8e277e65a85b55cfd90f8def707cfd4b026cc32d792438",
    "sha256" + debug_suffix: "265fabfd28d1c724e69207733d527c981955b89c5b4896ef54e145e9dfc40e39",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "8dfcd9845438e2d0ef28af8b8d73527ed92767349dd775a4002fff59eaf1a22b",
    "sha256" + debug_suffix: "5b6573b603fdfd3565a90345a281f32fa468f27fb87bdfa8dab9d5110e3bc79c",
  ],
  "kernels_optimized": [
    "sha256": "50485014ebffd0594b6b0c6190c218da34426633039fa8ff2dd4a22a023bfdd6",
    "sha256" + debug_suffix: "454112bcf45416241a6856bdefaa6c3ae94b47ec5fdc14be03c44273c129aa9c",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "2636df6b7bd9e2bb0d03a99c87ade1e70c28bf4e5f36db856c4d3eb99cf58479",
    "sha256" + debug_suffix: "81c936b3600461949e6b61c727b280e31b7ddb71c1901649e74571d0a15970f0",
  ],
  "kernels_torchao": [
    "sha256": "c07ac49c6386ed9a73f0c9829d69373b13cca3feab67e01393f029eaeae39bd5",
    "sha256" + debug_suffix: "17bbbb337eb23bbe08f19b78b09351d54f1d536dcc14b0ef4ddf4f5ff324aeb9",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "f25f15e91cb94551bcd4fa79698ec0e63e5f475f2a35d3ab3ae978ab4e33585a",
    "sha256" + debug_suffix: "95475530b7c195f89db1d497515288ac684173f5176c2b29fcd3100340baf055",
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
