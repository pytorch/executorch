// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250711"
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
    "sha256": "14ee5c96ae9cf8016063545e09f3c926aa26da096cdab73124f4a1ee4ab35589",
    "sha256" + debug_suffix: "7083e60fb56f3b318e990f715862ca851f4b22b6f159918e7f90e3e1e6fbfe33",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "ce6a5a3eb978d199972f98b164a36e9a21dcf51ebed6071494214f5d33f184f7",
    "sha256" + debug_suffix: "bc290088a87c1d72d9ddaa432060e6e1497bbf075664e267373ae9215bd73c53",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "cfad455d45e044da15694570e264b67893f9dafbdb51909496961089da006bc2",
    "sha256" + debug_suffix: "b328bac8420de45006efe3d03d523dc08c536e5ac21b925fac8674a24f664bad",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "881d8e2fc3c17b5f8a4473623f9ee9b303b28d5e0cac5bc2fb745b5dee1a9546",
    "sha256" + debug_suffix: "4986e347f8a92cc51cd302b896e63e52bc9a025e79f95b0377ac3d7374927ccf",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_llm": [
    "sha256": "cce502e51de89f2a0d5aefd17ed35aba29c79de14c80d66b277722ec3518a337",
    "sha256" + debug_suffix: "5d72696b30bcef0331cb44519bdb878b1cfc8f8632d2f1ec7a52e5e412466dd3",
  ],
  "kernels_optimized": [
    "sha256": "0658dc0081d2f3d88028446b36ed6867c394bb6289f859c043cf4e0cf1ade3f7",
    "sha256" + debug_suffix: "d73c75ba04e5eb320ab4f4f3d4a77ecbb405ef4a1afac9e3fa3a34b9218fa460",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "6f768bceeaf7d61badfd31237f6fa1baa2503d6948ace49a23da4314daaad540",
    "sha256" + debug_suffix: "60b0cc40a5350eeb748a6fe10f80a159274d9f18072c14c5ba78629aad26e481",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "359b5889767bb60bf1056d6be93388169d5e602950583c9013e6400478136ee1",
    "sha256" + debug_suffix: "9666087ecb6ceea60e7fef9ad76a985da143b13a3244911ce6fe7a0724417de2",
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
