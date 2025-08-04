// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250804"
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
    "sha256": "3ec7f1f66362aa2a458f162f9ba48305c09600417c48e379c907f8da41e31c29",
    "sha256" + debug_suffix: "9c85b05e4ef6eb693993aff61f7cff19d425cc022966791fdb98df8e8bffb37a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "6c40b93cf9c127dd0688b9e4fff2463c01396157353363a7586823cd801a1f3b",
    "sha256" + debug_suffix: "964ef8d7a605ab035a245619b114b460580f71d5627b27fc27c84d862b4b9773",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "4a9e778cdef716cdad736075ed4e1213a542b8a6811adc459dcfc7ce78123109",
    "sha256" + debug_suffix: "12d9a6261df27cb80c1c3de298f2bc9da9ae419f46cf35bf7bee88aa426e4a05",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "3357bcfb2efe426436801682c31b153d90bfd1375ad13a46bacc9cf4539bfe6d",
    "sha256" + debug_suffix: "8154eb494e881c4a97daaf5c9fd327e5671938d687fd251af3068b80edfa6d1d",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "2bcc4ef11e9d5528f61206ccbbf8b81740da6121cba4e6962df56cb0b90435fa",
    "sha256" + debug_suffix: "51f0e43595a5570118b5638515652cf7626825d51bc7b998767c8fc47d2adafe",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "082c6067832e344ba9c6449f40f45dca929054d0dd9bb5ec5d1e22c1d70e8a04",
    "sha256" + debug_suffix: "f2a6d0a967a7ba2f100b07f14ad4ed51ff926db0fdb8387a42580e6e9831dcbd",
  ],
  "kernels_optimized": [
    "sha256": "21e6ce2dabcbf6ee8f9886f50a3f1aa3f9de8cf69452f73810b94c57d00699bc",
    "sha256" + debug_suffix: "7ebc3a91eff8a24fedbc7166e7b8e7a65d327aab31e9552c82530d1c69a3592a",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "cc33aaeb3cec3b220998d1083e1ec1b0f2ab02b72e2fab8841f4951b30fd75e7",
    "sha256" + debug_suffix: "d5d113efac5a76d236129a0ea0e39f2b875275c82a8ee105a40e0c858ff0cf73",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "4f7d0a758e439a5eaca7649878cd9235ee41a3e82d295a648ebe4a18478f68b6",
    "sha256" + debug_suffix: "476bde829639cf76695e86c8fadc54d3d77c22ec09f2ae4df626d682e009dace",
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
