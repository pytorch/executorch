// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260411"
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
    "sha256": "ca4e03629ec3f53a86e1dbae3c58b741606567212a6f2d2a83b77cfb7fb54a58",
    "sha256" + debug_suffix: "c4de2f0b7735492175e110fe1958aa89c228b5da792e8201581f2e684e2805da",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "3edd4d27d627e51224b758c0b5d1d8a53a028efb1061b188c5bba8abd7f9d0df",
    "sha256" + debug_suffix: "080c9cc8e1853368d54aaa180a522dd495a756c880539680df273e86a7957aac",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "213e659d6290356bf704ca2cb0613378d20431b0bc85808223764569db68b58d",
    "sha256" + debug_suffix: "2b212aea1607fb8cfb35327f365e5fa59895024288d47419ac615a79ccccec81",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "37fe2a7c0c51d607c0a4bb653f05541c2f0c88241565e85dfcecedf234e47935",
    "sha256" + debug_suffix: "ffdb0afa4a7380097b39b90d058a5d35a961a14b850e5bdb009d1c851c5a9269",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "b32fd06daaa8e3db3d08bdf3ddda4819e2b910b0cba097aff47b6b26f1d87cbb",
    "sha256" + debug_suffix: "b9bc87dcfd03bc516104b31a20f794ca43a6a195c501e165e6395a5c0df0caf1",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "dfb0a3fd11fbc278ce8d57ac7a11f5a9aef565aad16398d4f90e56e366f1caf0",
    "sha256" + debug_suffix: "07c887aee35edbae296b6962c9f5d7e1708bc6409817204d582bc60de3db9029",
  ],
  "kernels_optimized": [
    "sha256": "e5832b2c3b721be9e1e5bd0fedaba20e4f3a9c52437de850aa1cd61d72d3ac12",
    "sha256" + debug_suffix: "db0fa63e2fcae2bc499e701234a6f2524fd29efce5b0e582f12b9532c0102bc9",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "09506aa1efc77000677ba9149bf0f58a1651679bfa9d563d2319c9c4011da89d",
    "sha256" + debug_suffix: "2c2d5b8b6d2524b34d76c182b6e75a940ec940743b683444a6371fb19056f18d",
  ],
  "kernels_torchao": [
    "sha256": "77319269232044b0eb6a41843fd54537dd858637952a19309e2902b26582b2ed",
    "sha256" + debug_suffix: "de024f8c10b6822f269a7c823d5d32ad03021c054a24ce4de396188f1c3dc6a3",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "cf5e937c8af6547bc03e82febb2526a4702225b6a610b3eb90d4123e9868ef58",
    "sha256" + debug_suffix: "c15adb0d8722aa3c6ab66bb113161f8e9a3d20875baeade59755d697b0ccb01b",
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
