// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260214"
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
    "sha256": "f073b2edafee83ba86a2318422853a8609bcdf9c3f6e642dc6627c6b895f8497",
    "sha256" + debug_suffix: "751fc863c85ebac5bc4cd14858049f79ed99b0d196f717feb47b8fed41573a74",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "56ff73cecd9adeb1c90e41e6e54c74a5e07f5c6a5df3095666b456a67b0ebadf",
    "sha256" + debug_suffix: "07f886a137acbe81bbeb3984502660220dd40f3daefbeda0762619784d21b8df",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "acde853a3691fa2921f97953413e958eb2e05ee28176cef2729b037dd6218a5c",
    "sha256" + debug_suffix: "e276e9965b0c27e9e64a4c783273e5414685c5a3ad1e0184fbbe3816d412063f",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "b6b88cb6af20dbfc341d3925da5806b3ac22948caf295b3e7508809d245f3228",
    "sha256" + debug_suffix: "5247942a49d959f5ea70d073fc546e2aafbbdc7176cf9a8158a8bc66f16e03db",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "eb5718d4733a142320e25a188dd4392ed9a87bb620b1d6f94ee3bc59eedd7af0",
    "sha256" + debug_suffix: "ea7936048cd164a29a13c942039a4900e98be299fb6ae2d0c1ff0a5453d0a8f5",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "5561c7564db703122c6eb87fa3ec4c52c1b7eb8b80f4000c859729a1832be299",
    "sha256" + debug_suffix: "ee6e9f7640f2d4d89124141dbadefbc074d16f5f90e6308372f43e006f38b8fa",
  ],
  "kernels_optimized": [
    "sha256": "0f52cb09e121839c9d88088ee34ef7a7fe6a5009e65005607539ece347d64e74",
    "sha256" + debug_suffix: "ddac8575d97fa966388cd170f203bceeeb51ba157d55d69fa2a54107fa3a7f12",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "098a904f5748ec80e17fb9e396e6361beabe0876df3c75b212209098e836fe81",
    "sha256" + debug_suffix: "51355f3a9bc38495da2a3e5ed7c0acaf337646a5fb92ac7adb6c107fb2f58df4",
  ],
  "kernels_torchao": [
    "sha256": "a7b4490334de593ec14a861b9f79b8caf4b9d9f66f665e0d4233c15b2507f233",
    "sha256" + debug_suffix: "a5256f4721b28a37ff628d7302275fea697f5ae2b7f038315cc18775dcabd0f8",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "b5de1083d2c270dd9e0e5b14d81b343c6739b65db10ed74c9ac00d5c79dab118",
    "sha256" + debug_suffix: "7f40905ca3f0eb64b0c198e4ecf97c83764ee5671e711673b968cab075a1e5e1",
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
