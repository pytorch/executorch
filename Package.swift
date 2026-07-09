// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260709"
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
    "sha256": "959f85022767fa960e9b5a652af7e9937954aaff151ca5bac5e70f9ffae2c8d1",
    "sha256" + debug_suffix: "1fe2fc73d7a28cf3cb1fba108d20797e47e01e35587ac042d0276d6e360d8cec",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "bf8494d6a56a6607cd36a3e4b865f1c4e218f45dddea97972b87cb64c0fddaeb",
    "sha256" + debug_suffix: "605343916e2530aef228d1a9b5c2dce4d7ac49753f5c95fb1c75dea4d39795cf",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8544cef858bf4d30522ccb686d0f252f10657a2615f7965ba47cee5893397d26",
    "sha256" + debug_suffix: "61cb0427503f6426af5924b00c899feafce644db10d9ff494ff22e5feb6b87c5",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "36456cd3af2245bc765a4c20f2c69214bfe223e212631ff9513dde5b1cbf4188",
    "sha256" + debug_suffix: "349a943a0a63c63b81d3e30a0efaffe85d8967e10da7d5e60838c6d8224f0723",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "83fcd6266bef7664b85acac50bd199bd4ba82f260ca815f2f47abbae7de266a2",
    "sha256" + debug_suffix: "21c1e728a6e10e9296e5b32a6a29a2f004d03b806a179a582e3494d03a709436",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "0783f39a0b629920dd3a1a34a3e22ba11e5658c3258d00854e298ebd3ec66259",
    "sha256" + debug_suffix: "bf9b42e0ef0c75552ca78906fc2121ff38a3bf769756faabe405b777e2df7ba5",
  ],
  "kernels_optimized": [
    "sha256": "c1c7518965be6afb7129c6cf162abda19fe13203d4a1be47b8803ea0490e640f",
    "sha256" + debug_suffix: "5ba190ca77301cf604a3fde43b70d35a416f0bc0c66070fc88fd15ea3f3789a1",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "c783e998db29db8a0cc0436f6e272722d5cea9adea62a323a27b0851fd8500cf",
    "sha256" + debug_suffix: "922132a9c410ee870e6bf46f7ffb65a5ef04a403fcab61421bedf6243b3bb9c1",
  ],
  "kernels_torchao": [
    "sha256": "c04174f0c344e70168fe818f551a2cdb9ab75de642ba46db02e24e7cb5fb289b",
    "sha256" + debug_suffix: "fc9f97754dba283b1bdc2ca347834955d9c4d5b0a01e9a2253af9ab29596d1fb",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "b40a49bceee1168416eb6f8a13f194ed3dabdef61cc0e80e3a63e311f55cf8dc",
    "sha256" + debug_suffix: "d4f9cc679ae1e0c2e8767efb546858c853c4f2dc675ba70fd9af39f97799fdcd",
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
