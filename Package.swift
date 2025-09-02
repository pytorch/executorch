// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250902"
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
    "sha256": "3284c741b1d802d2a5228bf35a03e204b3911e50c5e1fa8528f968fe274283de",
    "sha256" + debug_suffix: "1954de90abba4f4d24a8a820b1c43fee76bb2cae7be6eff1fff3cf4c4611a715",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1da3873b879a40449645337bbdab35f2579544790f2b2ae5f0a40bea1910aaa4",
    "sha256" + debug_suffix: "fcf8b9b94b07a82585ffd2b34a765247b0b18e304702a265d48543bc511d22be",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1fa1033865f08af8ab56f57ff23e1edbbaaa4a6a41cf527582d517754792ab61",
    "sha256" + debug_suffix: "c456fc474d266c1bf9d065cc7173ba586482d8cc547a13394f664c3feb1cb190",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0cdecd3acc8fee00aa7cb283b1dbc46f62d7d196096e1b110c01698952cb0725",
    "sha256" + debug_suffix: "f8652065f812d65a441e32cdabf31ec3b657ebe3fbe1939d6f4df0f3c9318593",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "66ae636fe1864e25f148b031114c6d757134aef86bb553d058797655930045bf",
    "sha256" + debug_suffix: "3e6f06d7364fb165aa4139e26e077bf11a164c54a8cc5323fc3de65dea3f46b2",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "9959087354070b728b2004b92e6e137639ea2646bd8aa866b1d1d61a631717a7",
    "sha256" + debug_suffix: "02d3229f06f312c826a47b1b6b178fdf51163f2bdac5ae3e2e09d053f0e5df3a",
  ],
  "kernels_optimized": [
    "sha256": "536252dca6a20becef9cf06d556c0c3d9bd4e80e5ef2f761eeb35f9554b2805b",
    "sha256" + debug_suffix: "41a54e732887b7e493494a25d10b4b693934a00613c1d3438d2938501427f748",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "34c0e420e260cc2b36992e1732c670834e24dc164b98f36d17cac49a70d02bce",
    "sha256" + debug_suffix: "96002b9f15ef57975ee3200fc53abda85392c8c72a92daf88788ef929af90dc4",
  ],
  "kernels_torchao": [
    "sha256": "f477800f707f3d7c5e8ae5ccc0d1aa0cde3b4b8d95d5ced7398fc8c1df1d85aa",
    "sha256" + debug_suffix: "0f222e47526ba743248637edd5acaabef3a529c95c391e420c028abe663c2646",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "22959163abd50a3e91d088a5b3f4748aa8fd9c28f7c36eb9ac07bd412a25649c",
    "sha256" + debug_suffix: "eb5e4b88ec73af2766d2e9f6fb7bf48914ae1758c76d494322c583214fe60868",
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
