// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251105"
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
    "sha256": "e7b6f001e331e57e2016416ce331b382a1d8e95ead2b6a0eff985ec30bd1ea09",
    "sha256" + debug_suffix: "3a49c6ef4436f8f1d30a93e6c79a24f89798a4dc21f2fcc716130f1d0546543a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5d88c87b83aa39de097d1c70d26c1714fec08359a3f070d2d95c2bbe6936426d",
    "sha256" + debug_suffix: "6c60f875eb578241436f374b43a1179f0d6a93c82ff3aaacde3d4f76cc990d0c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e1d4bd32f2dcaeadee63b9af6c5c5cdc268d379564d4f2e78402b395a1d04788",
    "sha256" + debug_suffix: "da2c40f7ff2e52cb7fc13a627eb6618c8f16e3aec34504dff8b3de36524187cf",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d183093366577db11ce33f631deda1573b005e93e63fc9e6605d6f3fa5299e2d",
    "sha256" + debug_suffix: "0f1fe0c99bbfe354d8d913c42ba323c01cf99ab68289c6b158142cfa503ee769",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "0a1a757b8c2fbb7dc9b43810e7cdaa16d9f05c3b6c4707f213a1788b5278d43e",
    "sha256" + debug_suffix: "eadadd7fa12572f1507807477225ba00aa10f60975daf5b3561e75ebc8a5a445",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "3fb481832cc7dcccde52aded6c72053c1e65a751ab9b50b2e64329bfd83b420b",
    "sha256" + debug_suffix: "32eeb756a68a8bc5ae83a1c2dcfd963ed4846e770585b226f745361e4d240e99",
  ],
  "kernels_optimized": [
    "sha256": "b808685289ab887f0dd7cfd7f3fd7f415874346119c8a3e329f58e25d2a7e7a5",
    "sha256" + debug_suffix: "e74a775afefe7ad61fecddbb48fd8bb24168c8f992878b6b8a3cd0877765a2d6",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "37d60a866f89be8d8b038f36a73d0e56da14d041904f31595e7bd6896afc6a8b",
    "sha256" + debug_suffix: "ed628c1ac2f4745759ac976300dd2a33117bc50871453c96b26996d03d6ba247",
  ],
  "kernels_torchao": [
    "sha256": "b37ce82d88291e513ca434f781820c24d5576f96cbb1540117aa0bd30f148dca",
    "sha256" + debug_suffix: "1e11e5731240d5476e14eb3f5056d917b012407e5e03a6014fcdc27fecfcf5bc",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "4dee857bae3314acaddbec3382f50e42dd8eca993e45d2583bc19db5b9562845",
    "sha256" + debug_suffix: "2934ff00b55f441f2533b69da173bf8bf76adca6c1a2a65ce27e372e84051537",
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
