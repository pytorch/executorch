// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250608"
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
    "sha256": "fd335ca363b2d05d3511a4ffa3b2f88c3e31b79ea7cf7e3f26da4c0093bf7b6f",
    "sha256" + debug_suffix: "80e92a5ee11f2c996506fa8725da81111ea2ddf8baa6d905d3f8a0508c54b033",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "12d31bd867c2d0461ab12c31d9fef80733dda17e34a4f23f9f999058d37aaacc",
    "sha256" + debug_suffix: "aa9f4d0442f24318757bbee43432b75eda179265ade344fe5e3182a5c368e290",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b53e2e4954d877bd6e952693bae17ccd7bf4921737a02ef2bf6a75e54963d86f",
    "sha256" + debug_suffix: "3211ab742277b9ad2ba748106ed17b69cb708ee5e2d2c7612ea4d8069e7767a1",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "ae4def1c84f50f5385fb819202f8a7b82b85b86d120b7cea7aa99c7413db375c",
    "sha256" + debug_suffix: "efc23f8c88a4a230d40afb68ea14a532abb8fd05f41678e666d1d52172ac72c5",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "f18d32ea2865ffddb54b88332b9e46f11bc61f6283f627b309ab81690ee63127",
    "sha256" + debug_suffix: "b145152f37cba991ec2cfd94f2b9adf0a008a6b8ce4930e992650fc30cf062ec",
  ],
  "kernels_optimized": [
    "sha256": "92fb2ab84277c519104317e86458d81815db34285530bcb0d6beacb9c6a4e3fb",
    "sha256" + debug_suffix: "f14199664a7e38d5d44dfdb082f17901a34046a38a392110ca2b04096722b427",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "21a29532fe69524921d087ea9b83c056e2ce61cb23202364a3b02e4105ad8d3c",
    "sha256" + debug_suffix: "e616aa61a727eaeba4273d82215c8cd64995de6c71d1f23b8e3f824780f1eedc",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "7284320fc8c24eac0d03bb2adc131080d1bff125222f55edd7281e576f6897cc",
    "sha256" + debug_suffix: "f51fab4b4fd9162e99d55a5ddd4b6e700d53cc80052cd23b95ed2ca425c13f57",
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
