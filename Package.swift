// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250811"
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
    "sha256": "42c032561131be28a63ebd63adee8bf0cac4a8b7c99061e69d5b5ccd5c5a227c",
    "sha256" + debug_suffix: "a0b19d4b4984f1e013e0edeb4b67cf716a2b621b07e2f323420ed175716451ae",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7feeab458d9279af1d8baf5e088164f46ba4d569365d64f880ee4b9444a681ba",
    "sha256" + debug_suffix: "3fe684bef30311ff4792c29ba81772f5b037fa635538ab8bb4814744e169500b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "2b5d3a687d0ef2d9ef56a95238101302dadfa3fa40c75f1cdd165618ba63ef09",
    "sha256" + debug_suffix: "f8d49d3f914f6674df9e61ca38992692886022da2afcb7a052f3266105da9a53",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e6d289ad618d607de0c253429c0667f6d7361d3c76d0f49609fab844e7d7df64",
    "sha256" + debug_suffix: "47441fb45d441bcd885b7e7aed29a6af7794500c46829a210ef9396f324be418",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "1faa6af60b8b2a2829072dfdc6211dafeb266917b062485b35ebd14cbc0cad34",
    "sha256" + debug_suffix: "7108a3d59d86010e131c0bd5fd8e16c6712d088ccf57a76a9a535e377e7bcf24",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c178a88d0e2ae04e1a26bbb214bff8dd63004824adaf6d842228890b7e881e09",
    "sha256" + debug_suffix: "f86a8ad29ba7b8d3daf8c56033ace5fd4d437f2b1a22a3092c1dffb1b349f495",
  ],
  "kernels_optimized": [
    "sha256": "db5b03510eb5a458b0311774352109485d0ecd0dfbbb219b1948f0bd8b6cf47c",
    "sha256" + debug_suffix: "845e0028b8a7904f15ac3136b4615ef89750d20938752deeacd2469f470633d5",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "aa0168072f119ef15ed609744198f534c952e0edf1ffa5512a6fd9b77ebcb446",
    "sha256" + debug_suffix: "651ab72b0083f2ccc08a6ca13f39083cb8ba57ea062473d8fab36923abb7aa01",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "2a14bbc431405f77ebcffb1a672ec21ed2b306ccf18641d507b7132d1ece8947",
    "sha256" + debug_suffix: "505f701fece40af62579900ff5df4840f70692121c5104e5d53257d4d4e6b597",
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
