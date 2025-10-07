// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251007"
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
    "sha256": "a3d2f342e5c4b5c5da414c8372636c6e6dd799175c0b441928222a8c2bc5f566",
    "sha256" + debug_suffix: "ddebbe2ac6b3f0b703ab0d2837b8bf790d0f3afc55dcf505f3ff3824b7fd861f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "263db0a761b98b6dc1d8fd15714141595de09af8b02403920bb689a7b7bc975f",
    "sha256" + debug_suffix: "46ccfeba98fd65d2314de61d4e8ba56be52c68b66e1b2d0922d19ef912c3fd91",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "4cb722d56349fed472fc1614c0353146077f5d5b5290799fa244fbf71ad125e5",
    "sha256" + debug_suffix: "85256adab0df9659191093bc0efc61e9a41863e8791d7e62618158b35a18c0a4",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0775ae3f3ec2a8dfc8a92960706f7ed5b74b7b291254f903bfb2cedeaa37cdf6",
    "sha256" + debug_suffix: "cf16b8b41d8960452aefa1850d47e274aa4949822422aa25a528520fdd9e217a",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "16552c4219fd042621fde58ec1ee8640f7e06f100d1b9796a7ef5971bdd13b17",
    "sha256" + debug_suffix: "b4d87ac9c6916859e97d99fa6fc7fcab9b20a4653b60f234bae5a61dd0f2bf9e",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "e5b4c0ea3963d491506d715a2b7d9e964845fe5e1e16c5f75a180ee6622b3c44",
    "sha256" + debug_suffix: "cc12f73c9c9c294dfd93dabaff56f1f43ef14c1e37987c9e6ccb72a42d42ff0c",
  ],
  "kernels_optimized": [
    "sha256": "1da700c8a2782ad5a1eac1e9ab7aa6c24a954f3967518a62447bc66e9e415ddb",
    "sha256" + debug_suffix: "48ba90bea2227fdf21b23fcda107a40d2f15e5f2e5a9b30e7f14cf9e6be0ec3d",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b924e4716ad606a04675e8bb1ca4e9817bfa5e13f9e63ee3f6f057ce8d675992",
    "sha256" + debug_suffix: "cac2c79a309559ffe1739c11c97d65e135838ea80d57c02dc090bc9a4cb46e97",
  ],
  "kernels_torchao": [
    "sha256": "f64040c8d08c49ab426c8775a1f978f854753383ec560db68b78b7fcb2b8aa11",
    "sha256" + debug_suffix: "f2f40a5edfcec350e9e3797a5dc6dc831e96977032d08a995558e4aba69d8857",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "7233881b37d38693e8ea2c4b6b309ab6b427640f7fe254fc126f4cb42b545e0a",
    "sha256" + debug_suffix: "7be0f2f1c3bdf2e5c98e313ad962597cc60d7d14484f882975fe8ae9573c316c",
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
