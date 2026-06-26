// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260626"
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
    "sha256": "93ab9dddf24e13d4489014a25cb60c182866fcaba8f7f520b3be67faefeb5fa3",
    "sha256" + debug_suffix: "a54134aa326b1bfd639fe2cc920453adb16f18a9fb9835c8de600bfda1657fc1",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "650394a310aa48ff0a7ea5c668b4b3cdc58fa26615657a010e96665f6fd4683e",
    "sha256" + debug_suffix: "e5ad76dcf4dd77edb7ffec653c9453a3d4ef4cb48b087246bed79ac5610b7573",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "0c18d7753ed3092e7cb2684f39e7319ce3e248bbfb09020271a0f29b4b276a5c",
    "sha256" + debug_suffix: "5023a5ce859aab5c6ae1ef1a49196e48d653d9ddd5a3a58d25e2ff1d960a1bc5",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "ffa3235093db354b96f0f96b232b45ef8951dcff81240055ce504c2653547729",
    "sha256" + debug_suffix: "a1321282ed3752b86d1f64ce63fd8f1245d3439497efaa11b40f5f7cc6bbee0b",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e1d5c0386f9ead1b5f548e470f48fd125acab2ccee639643a328b69d7e4a2a65",
    "sha256" + debug_suffix: "59d3209edb4f7dbc26152b000b87f07cc15c429d8ed6040e365ac6e802615a24",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "31c9a01e6c695df2254c9c77c855531d37c691673b05cd9bc75196c4b20571b7",
    "sha256" + debug_suffix: "4f9bea5bacb733531a7fdaaef0c2030a15ea1cb56caeecdbba5db7cb127261f2",
  ],
  "kernels_optimized": [
    "sha256": "0ab16e72cc136e227a979869a03d5728f948d8fc0185d0ccc64bd3ebc586a463",
    "sha256" + debug_suffix: "ac07d1c24dbc88c920b4730e3faa5fca1347a6a7fcf301de0bc80a2d0ace0f83",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "f9c40d46813d09de1a3d568764d2d5b3c04b4934d5155d3a3d9b70ef8abb93e2",
    "sha256" + debug_suffix: "2b27cc7904cdee66cea7c8cb03b4003b979433cc3ef4cc2a94ee8ffc435c0fe1",
  ],
  "kernels_torchao": [
    "sha256": "173b571368f525804280d956befc933e34bffa7b3cb9214ce737ecc63ebe9221",
    "sha256" + debug_suffix: "5352674d6dc4360d7754721098b80051119173b42bd608460cfcf8bc831469ab",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "28a37f16965e9fc8c63962160fd822b4edea62147e6ff9eadf1d8faa007f1602",
    "sha256" + debug_suffix: "e3410516a9336f531bfa6a683f0063fab5b6788a7a06dc6e309cb214ebaf26f3",
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
