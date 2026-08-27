// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260827"
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
    "sha256": "dbb4f6ee9e90cdcb2ffc96e2b9925d320c511a120dee5c53794ede15daa37bdb",
    "sha256" + debug_suffix: "26e7c738eda433c0a32351cd8f93f52c84b807696d7b77234aa9af389a8c9ec1",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "18685674729751d345d65265461c00287f9997cc0d822543b97a5d30fe2bcd27",
    "sha256" + debug_suffix: "58486f570d3aee2aa3433bf86a1713f92931b1b1d408e37dc2cb42f42ee8d96a",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "6e01d14bc3d9be7ed9f7d02ebc4e803de4357fb501bd36408237c70e1c0f0eb5",
    "sha256" + debug_suffix: "71d462be1c47a04f5c1c67c0f9def2689ee7941d613ee9246a14e22b4b5fb125",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "f030612c60a4ad297c330b94c699fc5cc7a645e10df58b438bbc44f1d1e59273",
    "sha256" + debug_suffix: "d1f8e5759d0cdc538989e0d3ca91ddfc1eeb9deb99604036231badb79681ab46",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "5fd1a80e69af52a80ced409a0443733225d50e218417f8866785d5e711dfac43",
    "sha256" + debug_suffix: "6f24a1b341a78beb99b8645c28a3f2d115f9f2ebfeee421f5b4397d583c9122d",
  ],
  "kernels_optimized": [
    "sha256": "807abf8af92f4e88ae5f0dbb8102c1ea871b939b81d25e821579ef0b121a09a5",
    "sha256" + debug_suffix: "8af57d2f4e12b84d1e68a6a0c612e5d4f75fbf6423afcb1784693ac793ba003c",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b1c03533d96dd77c418de05e7432afa82a5d8bf23946dfe0a702405f1ad61711",
    "sha256" + debug_suffix: "c7c26522f1414aabb316f9b5dd2fceb99b529037c498f16990b84b906d9b45db",
  ],
  "kernels_torchao": [
    "sha256": "c5f1ee25922efb10d1b9c0964b7b5755ac4d70eb645940702a6653fb0a8e6b1d",
    "sha256" + debug_suffix: "cac24b61de029b01138a68534550b368a7dcb1f6f3b1b8a48973c6ebda5f62b1",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "1e17022c1db9c451b3b1ca872ca2dda093eab54e4d6be2b531cd26e2ffb09bac",
    "sha256" + debug_suffix: "929e798b4c4c9bbd78316e846ac00fd03a62ba5e9ba97b5563404c2f8aca33a5",
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
