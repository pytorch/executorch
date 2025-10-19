// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251019"
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
    "sha256": "d76f757f6cd1ce7b36a8a5641a52bd67db4059c47046d6ee06aa679973813c5a",
    "sha256" + debug_suffix: "9734b1244a591000bf40fb9995ec528eb4ee1221c40a74c2b84437c2d84a9aeb",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "bd9f5f6fa15e59c8139973f511c6be4e2d6d40c2b72d5e26df4622f1ab91e2e2",
    "sha256" + debug_suffix: "4a467ee80943430a453977d5558a6df7a7c083f186a06a56edad496f30deda3b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "88be1f3a3e5379f145c2a85028d5f1ca8df520fb3811b87ac468ec17e3e4502d",
    "sha256" + debug_suffix: "e0d3509cb90286bbb165b99d6db08ca83afb00401f96bbd8c3568f367ea7ec57",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "04099c6ca4e83a1cc69cc1965e51105f6e5154b71aed8efade31b56898d8a543",
    "sha256" + debug_suffix: "33f622edbb05cc08b3bedd8cd62135acfd0ff8a8862e8b6e87edb30693719c32",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e74aef7ee210d58c4a1c0a0db311b810043d625bd2fff582b11ac623a72db518",
    "sha256" + debug_suffix: "29be2e8b81ae3907ca90792afe20141236bf2f125de2e001ee4f37111e03a58b",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "d3f8b812b7efa9f6328f87379898030edce74272de5f908179e4823c3c524f20",
    "sha256" + debug_suffix: "e8c837159fa11f126dc2ee57f39cb8315e58322401092e343a5518ba6aeb4be0",
  ],
  "kernels_optimized": [
    "sha256": "a2a9f19a86c285d40ea5d927e06a20caf18cbde5b4ff4d6c3c093ac0dd33fd5b",
    "sha256" + debug_suffix: "2c09bfe15e8bc3bf8d805d5ef72b8868731f2e3675ae8290a5de3fe680e4cd56",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "48106bc2fc111c92929813edaf90ab3c8f5ffd3fd184282398529c39e3558cdc",
    "sha256" + debug_suffix: "8e87d57c0ec771426be19f55fdf5bf7ae062091734b6684505e2a514ebf0028f",
  ],
  "kernels_torchao": [
    "sha256": "a7b04befd73b5b23868fb88a003b4d30fd84637521b5983b90f7fbe45d895b98",
    "sha256" + debug_suffix: "24cf97f4d83b12f3ce9d23a2dfc55201cf9992221a3a4bb9abf201f2f089d705",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "9df1b7b40a550732b9590c41f88171b31d6e5f5bd26e9180f6d90afb8738676b",
    "sha256" + debug_suffix: "f5017c5f5de75e3f233aa0c4736cfec072976ab6fcd1e2b87e666a40e7b5e7ef",
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
