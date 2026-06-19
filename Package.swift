// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260619"
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
    "sha256": "8a6ae61a50b82dc094c5f54e7a93e6c7c2ba68165f418203da96a5819147508f",
    "sha256" + debug_suffix: "9b780bd0a9835aaa4c23de9efcf9ee672eb73c4aae878e59ded468ae51a4043a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5738005b60c02fc7bb2144de6f60e5ad22e167e8dd782a527ec45c6fda238c86",
    "sha256" + debug_suffix: "ead0b2a34f5e5ffb2caed70e4ef93de3c26b6fb8bcef3f4e7485cbd9c7427aa8",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "546529e9db512e46a7e5d554860afebf56b22fd7d52dfeb7cc190f3ccddeeb1b",
    "sha256" + debug_suffix: "fe41c5107e199e1ae33bbea9aac0d57b67bfe8ad3a74268e3e149b300ceeab67",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f13dd371c7a72338b9ed93c35a96372ae995ea29e5c232a093abfc7c02b6edfb",
    "sha256" + debug_suffix: "bdb9c5f2e796514209e7a7025361aa3c3509caea2eff3faca54341650e7b56b4",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "516213313a5ca022b0743031ebf2d98641e8309b9c8ecb3c93612704031c2748",
    "sha256" + debug_suffix: "fe4648d1e0e331af78988ae46ca382a56b9deecbc9146d03b3f3b1cc81945742",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "8300058212e141e067adc7250ccf2c6582489155589f49a82a7ff18c6cb6d027",
    "sha256" + debug_suffix: "184c3b5a9e80646f2942b25f48b1ff25474a631e88cb832078c856064d5256c8",
  ],
  "kernels_optimized": [
    "sha256": "4246f49a7b0a1b88e70312e8c5c01ae09466465747f6da0844c715c3276a1e79",
    "sha256" + debug_suffix: "325c87c1afe321751a0e4577c0b3a22b5b446d70067731679b541bd60bcbbbea",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "6fab52cace78031189353bd90207addd79cc40e0b2693463d04a55ed6445e720",
    "sha256" + debug_suffix: "fa7f713c30ff9b682fa1e407ae0b0dee703b772fe1f2c771b4472d9652f131c3",
  ],
  "kernels_torchao": [
    "sha256": "59f80523622ee4217c872754a2617ad2186296e17ecadf2210586078334e4efb",
    "sha256" + debug_suffix: "d2a30c8414e0307689134469e084c37ce1c97991ca0a03844ebaedad73484790",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "234325f7ea3927272c79dcb067360f9d92691a7b1a4b717cdda66aedff009b1b",
    "sha256" + debug_suffix: "4660ab1d41126e921fb7fcf05c58f7fd62006fdbf5a26ea417b38673484c9081",
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
