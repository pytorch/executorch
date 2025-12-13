// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251213"
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
    "sha256": "79cc2fe91d3131b1172044bbcd2a3fac8bbbdff388edfbd69740a41abf12c9a8",
    "sha256" + debug_suffix: "2977230ee41cb3469b4d41d8b698c403dd07acd5be9fe97bf0d413df9f1352af",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "1f01a6841b63a902d786d98d26d4ac7e5d663497f193c3ef55c43bc922627746",
    "sha256" + debug_suffix: "14ee006eca1e2d068544c2bcf12eb6fd5f37dfda5f478f54d9a9c8ebdf666aff",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c03c7acd8628f8e012cbb1dfa3b9a5865754feaed42eb8daeccbc1a95df0a01b",
    "sha256" + debug_suffix: "861bcc21b0dc32ae75d5d34ab94086da44816f53fbffac81186880be5a9380bd",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "bfa617deb717b2b8210465be8da2f79f782a68afbbef6a13c627d5ba55638b42",
    "sha256" + debug_suffix: "723b72b4f56c0020eb40ff5d9605b23a1c78046c620555ee331787ff9886a524",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e8ee3a4ab7507aae7d19e6721f4733a07ed7a26cd742d983b19a96a7dd2e97fc",
    "sha256" + debug_suffix: "f7191fa79bd11112f1dc5c1a1e1c11d76835730b3248c22c172b3e6a638cf9a9",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "14ff78d3232e456bd7963bcbd2982729d91ed9b66ad7c0442ccb3bf84ba89428",
    "sha256" + debug_suffix: "de5c3a21708fac57b5b99eb78af7cbab1f804ef6df7a2af7da9526d562a12899",
  ],
  "kernels_optimized": [
    "sha256": "f053ea579e15481917fbce1b89c1a043996acb5ebc231ce588a43f134aff2030",
    "sha256" + debug_suffix: "452e20586b68229b1a4542add96727ff244bd7dba8ec06899ef1b207954a5279",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "381b2269c46a413c42166b9c15dafd53d059ef5ad057d7099bfc852ec5cf4e90",
    "sha256" + debug_suffix: "da85d9050b34504a835d86f24ae4a47c3ed03599f648b4deaa532e87b94ebf06",
  ],
  "kernels_torchao": [
    "sha256": "d5aad05c6bbf7943eb0256cf2ca0a17e92a288474411ff3aa9f73e3a46e51a9f",
    "sha256" + debug_suffix: "5d3530a69a0e6ae0b2a36e18da205f676737cd8f32b749639d13b31af99d338e",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a6389d475b7a5dae0c2aa8672866e78f755573e1de5d4f50a021472f281465f6",
    "sha256" + debug_suffix: "5f103f0442485c0873ccc03aa1b26c2145c50101955713a1e4e8a14f793a26d3",
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
