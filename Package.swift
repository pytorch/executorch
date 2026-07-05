// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260705"
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
    "sha256": "21ea3b9a045d96a7abc9a847309e1cbfda1fbb78a8bed13bb2b42a9716f3cbbc",
    "sha256" + debug_suffix: "97d63aed8a1b7385f8957786c2e3fa01f045572c78602308588b7b1e4cc5ad2f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d0aae2f8cc4f7a568e690753cec6e2d7f58430ade06ad48c9fbea32fc8846cd4",
    "sha256" + debug_suffix: "a8489d171169ccf5f489a3ffc9c4a65f532ff57dd0e142c91f6e18ff8ec8a387",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "cb2bcbf95573a7db1e67264ca123148e3526cd8ecf2b0c0b234202fec530bf10",
    "sha256" + debug_suffix: "ec770eb40163cc34ab3e04ac2283595965eecddfd96c3c526e8005d93f869ac8",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "cdf3532a66b4228a0a39889640f8513dea9e5cecd1e20ffd2482534cac999324",
    "sha256" + debug_suffix: "22851fc95d8371301a7212006ad36a8c5d1c0566a9ab838b1b45f764b51640a9",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "2a755de75e755f4e5cfaa26694182477936d0b17eacc2b2a8961e481df28dcb0",
    "sha256" + debug_suffix: "a2574e36998b47915ac15944f85b3d873e0caf9125b9e2208198cb23d22f772f",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "3957360e0bdbbbebffc077f2fa74da2503d863b82da4f06e340c39daa6e3c3ee",
    "sha256" + debug_suffix: "5b608e6a7b92dd170701af9dc15fb896ba69dbf75d2784047fc0a4ced81dbea2",
  ],
  "kernels_optimized": [
    "sha256": "c6dc83911c3537bd8d17e029a6341d33faf95138ace1c9356379cc862eba2081",
    "sha256" + debug_suffix: "310206d27dddcde75c64ad7f5979cdbebe1b9652a437b140d754cbdeb0bf7a85",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "2e7ee82ed01f006ec47c555fbd823396cbe66650bf9d4417f58573aba6e49620",
    "sha256" + debug_suffix: "5b030a62cd097799ea1cb83a5c31fa2d77a5df0e22686c39277b93583e747a43",
  ],
  "kernels_torchao": [
    "sha256": "e1483f96ef523aa53d5a005d6aa3ce8c03bf9404716ae9529fac5fe80b25de76",
    "sha256" + debug_suffix: "b0cf287900662b171065bb09fbc6f456eb030f034318d038d9cfa4c5035feaae",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "7dc0b1ea8e7261e596af42e09cf7e17a7402f28a71e62ef6fcd4f6c8d3181a7b",
    "sha256" + debug_suffix: "3f7b9aa5a6820119fca34833b4ea2f01d6200252f4c4a9732096daae243ecce9",
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
