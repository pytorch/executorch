// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251210"
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
    "sha256": "bfdf20320e486620d3903c6021a279cc3eccaae5a3d2a507e734907b7c03aa58",
    "sha256" + debug_suffix: "4d003c59dfb75430b3f4bad91d3bb7bc94b97de4fee9b2a5342237e81c54f498",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "fb5dd1fe8728883064df9c88a2afe8c70651a881fd0afe930d483beae15abc9c",
    "sha256" + debug_suffix: "31a1eeb0f8ee39b59db5498dad641671098fb6bc884483d4649712a9dd3feef2",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7c9f665de2260d78cbdfd6ae32c0488729f261ff3c8eea2faea15dba6428a470",
    "sha256" + debug_suffix: "5b976e9037162839ebe905cc89bc5a6d1a56382fa5f5a040cb76f18219bb4426",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "c3194e404fce1b1baa20be34283d78aa9fa86d3e4aa21627b5fcf110e3d7e348",
    "sha256" + debug_suffix: "64382d410b36ba442a3c8935a85924458cccf15b50b2cf65458b9a85ec0e1a4e",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "fab054e96b576a0cc4c3016a0db39aa32782dfd63d96a55ddd808586cff7bd2e",
    "sha256" + debug_suffix: "ae1a17f2fb4087acbc6da8d5ff58f1dcba07b881191304a334c50f3e1527d8d1",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "09a40b6da425863a0d52a82d6a0eaed3cfad7e14a167e7a2f6e4ac0171e351cc",
    "sha256" + debug_suffix: "36a2a5e9c12d0b1f3808637d64a79fe0ebd2c732803585be7c7408787cce0a94",
  ],
  "kernels_optimized": [
    "sha256": "b1cfe11519ef1a5da4ba58b9a176465b210a3e1930672b720fac03a0fd589e24",
    "sha256" + debug_suffix: "4543b677b8fd389628cdff7e9755290867e57939e3a8d3155c4d934cc31021a5",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "70ff1da23412d25c9b295dd070ca7ab98b2ccefb7c8fe050e3e8b5ac2d7de474",
    "sha256" + debug_suffix: "05e24ee9a3c59e542421175176fd51c6d5b8cba243f715244ab066210605975c",
  ],
  "kernels_torchao": [
    "sha256": "77a0b9e2c5e034573dbef9a9b5d45f2eab5e4d437291a7b32a6a8aaf9886402e",
    "sha256" + debug_suffix: "9a01b5c7aedc697c4f02375aa5231a3ffb3e43084038411c0aa1bb4aa7b4e136",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "8dd5eab0e2f08b56d0bd8d35cd1aa44f5269faac30e391e241d256261bf0f635",
    "sha256" + debug_suffix: "3834167695f38b067d234524b7c37549ed9ac774ce4fdc7f1589b78600f179f7",
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
