// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260502"
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
    "sha256": "dfa8ca2c8f44602f278a345be5bea7fdda495065f556e0ab6b0ab1350e79c09b",
    "sha256" + debug_suffix: "7397029c48ee6c3698f3c24449d19700214038a436454c3f8a9e89d68f7aab05",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "89d453348e527e46540f56014514cf97410a232e17b96b3b4499073dcbf4ab08",
    "sha256" + debug_suffix: "57b101911a4a181233b8d0369cd0e5c6d033dd942e60f071c60636572b85d057",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "17c7503453be2374511dad7cf21e1f020dffe4e2a5b1f4ad4269723fb19d5c80",
    "sha256" + debug_suffix: "97a615bac87fca514e5f027cbeb92d1fae2500b6373c2bd0f509a5d38d0a1b97",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "1929d3d4a6baa0028297fb59d09575a2fe3d2d1e4fa108e20cb4273e2076edf8",
    "sha256" + debug_suffix: "da015535d727597772107004c0809db7370c4116df59f25c5376b1fceead4db2",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "83dc570d994221673a9e6716f28a043e65394f9b7580c1e15e5f2e17a94534d1",
    "sha256" + debug_suffix: "758c3c2078acb3d236cd1340970926f78b7bd5bb2a731e1fcfe810721841c478",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "1122e8e7d8cc05fc24aefcbb7cfb9ed71ee2dd0fc5be50f0f4eabdb5e4311c12",
    "sha256" + debug_suffix: "3313d41bbc699da37d24f4f6b324061c865cb44e4293d8105f2ca54ea30bc383",
  ],
  "kernels_optimized": [
    "sha256": "390e38d3e5357dabe7af3de7ff72c5bde2fbf6d4a4e6356038b92a70d954ca85",
    "sha256" + debug_suffix: "75cadaacd1935f8eafdcc774ddf12ade34429fcadf7420ff67a9c69dc6b7529e",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "91683e833c2918d278c6f2efc9ee5ff91ada8329a85081d642c9012fd4aba47b",
    "sha256" + debug_suffix: "ec5d975cb4f8eb1a63cbcb4b051c95defa3b98532bc92083f0b7cec8d8fcecd2",
  ],
  "kernels_torchao": [
    "sha256": "3a04f3c1889ef6b7ff62914cf5c453ec8dc3f915610ee7e089ab2b197fa86fdc",
    "sha256" + debug_suffix: "1ddebcc746a010dee10eca05683b3c204001007344e11bb390ebbb7a464bf03b",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "42e90866649171cae3eda80b7844a02341393d862c22588fc032954d1e298d4a",
    "sha256" + debug_suffix: "4804b9e53dbc58992aa891b23a70ea78d8b6b8576d088081c0366d31b16b59f2",
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
