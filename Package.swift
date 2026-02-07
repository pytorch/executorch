// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260207"
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
    "sha256": "66ac11ae705a0b19756bbb2497f8002cc6b7ce24fe245b9101b4ed8f9d703517",
    "sha256" + debug_suffix: "02b1e40369527b8d5b245f9a1e8d8601966f59cccf2018168c96fe6ac1080098",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "15d639aae5992c164d37d25bb9c3eaca14cc7fcecb5f3f3362f9f3b2b1a84782",
    "sha256" + debug_suffix: "627eb4e876e2e81ae41d606e467dae9c8ec8d1af45415b6228b3adc264c9bd60",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "836a56b4133815462c750a4eead2c2a141caf45a025d1d673ac4ce580052fadd",
    "sha256" + debug_suffix: "14bff943d84b27d69646f791c1012f89e3d796a1599b12a3f75ee672a54cd92b",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d968b073e6c517107b1b80c3016d7f1350ceeda7330447aa6e6283b1fe606aa1",
    "sha256" + debug_suffix: "a0e09d58cfe0b8cbfcd1d1b75d671cfce1d960ab7886c39c3a0886b18ab669c5",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e755d981126e5d3b572eb34e911226c5ce9d5042167c5cb09d0dfe19fa5716b9",
    "sha256" + debug_suffix: "0aa4b1c98b5e45402a7efc9bf836c0aa677f5f2984afe7184c0bff84bc457082",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "090e019a05b1f1ec5330cce882d76178f5232991f4bc655a66707416a9e8a477",
    "sha256" + debug_suffix: "ddde025e84b7c6bf709d740f769294b8a9f9fa5092cb9ea82d038d2c5406a197",
  ],
  "kernels_optimized": [
    "sha256": "8a2abc1c506adb018f31a6e948ff3b6f1b4fc700b99f712a413f8253a20bd7b7",
    "sha256" + debug_suffix: "421f3f13867380e6f7c7c94d1c991c1aa364cf34a7a8626c73cc819268e71602",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "2f139dc4704de839450c146d658911b49ecc7f3f073d65045ceec7f9877eb150",
    "sha256" + debug_suffix: "a4347b54ff368fbb2c9e7d17ad6a920a84f040cadedff6ba54d3d5688b136e10",
  ],
  "kernels_torchao": [
    "sha256": "220793fcef02c733a954cadd569f7a1955182a6336701c77968cef6989107717",
    "sha256" + debug_suffix: "4a587dbdc9703bb04f85db2a91deafff9070347de8eb50e4b4f845bec28f0774",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "7c14637fbf161868ea8bed0f57b098806cfa9c81f7a4e62c832fe2af5a92a241",
    "sha256" + debug_suffix: "537ad4dbb974af7e30be06688774e32ab6f25a2f4f334f64ecf0d859baf55b39",
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
