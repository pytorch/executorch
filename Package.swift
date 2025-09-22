// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.0.0"
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
    "sha256": "f323a1a4258e27a39604d5b42477c63598eb9a462f703689c6bc4c84d8eceb7b",
    "sha256" + debug_suffix: "0af75c7ed1d2311ff397d9893a4d01987502a70290ae9b4126c1f3c3e9d85f7a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "41cbd53ec8634a1e57bcc68865676b1fe189953ed89213a7757374e8df7106f0",
    "sha256" + debug_suffix: "c6386d35e2e9941ed27078d9dcd51e10136fcea38d0062e9028dbfadbb243555",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3454e82c563d1deed6376df7261d0982205800bacfe81064f2807a2dce421a5e",
    "sha256" + debug_suffix: "3133c3b6883eea8997d19807293aa8e67b95a932c78c9471a13a8ef9627b93fe",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0af88e92643743ff189a4ed54d846280081d59a9e78c49ebea9599fd9baf43b1",
    "sha256" + debug_suffix: "1227c8cb35898044d5c1a8f67b70cc257cb022455225a44714845b8990e4ff44",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "5bd8a82c6a22413589e7640d3146939ae145f29a1b2a6dee9c2bf4da853d6c1d",
    "sha256" + debug_suffix: "b9ccdb4d18b12b952a22f4d7ec557ac205f7dbe4092b8b8013bc9b70bd457b31",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "a16088f91ad527e783c5a71e16928c0564e5f81041e0fcab37264b4677495526",
    "sha256" + debug_suffix: "9c77e78a646f942396a1baed5966a63c51e45146d6f9db7811477e4c776ad675",
  ],
  "kernels_optimized": [
    "sha256": "1eb611fb0dceb8359119fbb50ede7965eba78e43735fce7dc729b1e8f6f8b126",
    "sha256" + debug_suffix: "47c7ce99d0d4c1d54e2790d3b5db73cd3d73762b350e28a307d46a6c23225e7f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "0f069bc479dbb43619662b4e023f82ee07c6977f3a2a2a47c56d4b0bf18c6328",
    "sha256" + debug_suffix: "0aace5c9d26c9d2e8914ffa01d1a9aff7f8fb2bde749539f08c19a1f5dd052b4",
  ],
  "kernels_torchao": [
    "sha256": "4fc68633cb3837bc978f31fdd0f24c260d18e13677e7f80afe83fd815805f5aa",
    "sha256" + debug_suffix: "81df7280c3d053cdcc3f7e2b8a37a8140c7015417c1fecbf2f7e9e75179e9cee",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "2a236e72e2574ba8b7e3e625ab7babdce62420113837aa9eb4b707bd84386cb8",
    "sha256" + debug_suffix: "39d21542d787eaeee2691f5ea031a63c89219764a60990c8f85e01cadc3337b0",
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
