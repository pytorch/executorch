// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250619"
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
    "sha256": "35b3c8354515b7742fb0170df8d4f7661edfa801e929fbc0ef30affac5001598",
    "sha256" + debug_suffix: "a2d002cd59ed21a579a5a60731dd7fcda988292b44805d90dbd28422793c2805",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8830f5a6c49c0cb64808fefa43b27eab8ede970cbc814bd86d64975ae8bea1b7",
    "sha256" + debug_suffix: "c055b6012cfb84e0726153a20b419d3b0fce4af76ba10164deda53d29a867707",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e358fc66219e9c75b3e80212723a27f2f695262febd84fe4f7e43775104e9e79",
    "sha256" + debug_suffix: "51ca9475263c194c2eb09821189d546a855ae134472aecf83d8a0dc821b91915",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f074a90d970431b7f4940b1ad929ecda52c562e743ecf393dd6515d59403a927",
    "sha256" + debug_suffix: "a586a2a2173265840cda95d0703549c8d22ab3d63319fe68c1fc3076bae56d52",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "7372d403a45a60da5f093b5e7fe769c46bb7994c6ca2b7595c28cd83a82862c8",
    "sha256" + debug_suffix: "140382e6beb69013f4748723420e0e9e5a2842446d3359f55baf08cc948b7cf0",
  ],
  "kernels_optimized": [
    "sha256": "6b638dec0ef0dcbd539f3bf5426d50ac132f87bc8e313f9c010152b425a254fe",
    "sha256" + debug_suffix: "990776c1922f049fb741f610c155a7985468d9a165cee8565b546e6c990173de",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "e11c03614bd5dae4dcf21b053949a5d5ed1f96e348b04f0acc9b91fd2989e0da",
    "sha256" + debug_suffix: "6875a5046c18f6ceccee80676efaed9cd0c31f095a636e18d3e55509bc73d3af",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "618012c8f0423b58fdb83f1ffb6729aca64d4c9e2030793214af4f1b7ae9de3a",
    "sha256" + debug_suffix: "0717c84d07f0fcd5ea1ca032d51d47bd859a481107de74d5882ab4812a2664ad",
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
