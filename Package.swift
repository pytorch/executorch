// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250618"
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
    "sha256": "5e92091743ebdd2628e3d7e77ac34fe3ee0173142e175816a3abbb4703603708",
    "sha256" + debug_suffix: "db9ef4409fd807d1f678f0b5e9b3616c6fc9f8135bb11bcb2ec9f39bf4902518",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "eeec74aee301fd58cf919865fcdad589fb18fb5846adc51c18216c63a0d2fbc1",
    "sha256" + debug_suffix: "100a126259b50344e85829ff6fdbe2742ba5123780854c120d40ee1e4bab9cf8",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1494b846222a9a3149057b78014f53afbeda1b2e3cbaa655a2d7af172e1a6976",
    "sha256" + debug_suffix: "21b09503e9db9c096c13a525341e4daf22915aad13a47704e4b777c2235aa1f1",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "001d3e60c67634074c8cc739267e80f804ae327f8e21a261915b8391ef918196",
    "sha256" + debug_suffix: "137ae91f3d3a967e73bda50c87f5a44537a10a938000e942a3c8712bee3a8034",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "76d5a5df9faeb62f93de11cd56a5d9cdc7e0adb1b1be9385f1e91c821df8bbfb",
    "sha256" + debug_suffix: "24219f085399c6c53a65523f0cb185edd09626e78ed23857ec99e0be7ccfec8c",
  ],
  "kernels_optimized": [
    "sha256": "8d74f6d4a96a74ce7747b3e8fb4c2235d0dc602cc6eeadf8b755adb3d6359592",
    "sha256" + debug_suffix: "194b2e70d168fc4a7743d4d63984df2b2a0a3d2cfeb07f9c23863a93ef25de32",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "4471443a538bd1257571c8ece50246dddbf39a924fea79e3fc299d0940a92deb",
    "sha256" + debug_suffix: "f0f8c635f798645f21588f83f7bc2912d83e9ed251afa67cf12a65049f5cde0d",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "d58d0d7da1c3bffacef5aba22f33bc572c57f0793829872997518b87295ac6fc",
    "sha256" + debug_suffix: "4b56a8a6e5b008a907b743d5e55fd52013fbd42095113145e290824265e7df61",
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
