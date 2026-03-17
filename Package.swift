// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260317"
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
    "sha256": "7232fc27bec953a1d3388195f4137b39dfe4b223f2d38eb7bff655756a607939",
    "sha256" + debug_suffix: "4cc28eb51d24cd47775e3bf6c4fd627497d55fb6c150ac20a831c71d9270caac",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7c57daf4596257594a998a10815d0d3091d3bed2341b927c657633d8e2ab2b49",
    "sha256" + debug_suffix: "b1061f34e952a7fb3482f54d9a97f45c42dc60d6e66f616b95ab4e0db533c87b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ed46bcc97a815c7befe2677d0ab29a676cffa660d5ceaa3665fe249b40d5e3ec",
    "sha256" + debug_suffix: "119b4af134744e8bc66d9fcd585b9abac4cac709a8a40228b73efabf5d40d270",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0474673ae9084d8189b43c4203a97b111e34e3e1d6e4870dd0140856dfe7b948",
    "sha256" + debug_suffix: "fda9bfb6c02edf6a2f182e7b3a0d271ea895d98601700e382904cf002083dd17",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "65b0f9ff69e3f5326b43ed3c70864596f0bdcfe4930dd44e09580c0c0ee6a318",
    "sha256" + debug_suffix: "ac2a9929b85ee48f908526b0a0514c99231aac23aa5b247400289439289c897e",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "86b8950ac18ffa2c1208f659c208354ec1fbc3b57edaa90c5269cc15572c33e0",
    "sha256" + debug_suffix: "26ab698b742f4df4d5229409817a6996dfe6447a5c7f45958251d929672f0ee7",
  ],
  "kernels_optimized": [
    "sha256": "280168a31221e75babfead06380c2ee3c3591abd9bae1b7580ac8eb060888b10",
    "sha256" + debug_suffix: "2b30b1fedb66ca4b96ba6708cd5e7061f16d058e1c97c00b811cd09f26d062ad",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9678a1b084cc86972e44d84ccad4437177a1357322933ec0434d526ce1eceb34",
    "sha256" + debug_suffix: "3fe572612656dd4ce046a759529a55b635efd18f2286548a8b816ee508c866b5",
  ],
  "kernels_torchao": [
    "sha256": "bf46025cc0be19f38934424cb4df20e7c7461d53af3b5ca11523f4bfd9207dcf",
    "sha256" + debug_suffix: "744a2f975374e8457f99a9769909405739c9bfc572a49ddd093e6e5a75619758",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "b6902be4cc8e0da7a7a3e8c338ac1c0128d0b881eb88ab1c5d0da7d16ac509cb",
    "sha256" + debug_suffix: "55a558dfc6fb798966b0b2a0cb68253b6518ae4bf13a18aabfd075aff73039c5",
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
