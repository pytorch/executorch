// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260423"
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
    "sha256": "c9a39a7bf541f9b2afb18017d151300a69967aa3aae791efcc649bed5a2b303c",
    "sha256" + debug_suffix: "fd5867a76174ebd6e2708a600e2ba5df1c51be2d68676703d00df11827d1e02c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a522022cf45b4dcc6fc7f26b7452662969d9bcce3154cedf6ba27055ce7eee5b",
    "sha256" + debug_suffix: "64d425d67d8eb84c11eb6a3239e80e20fea43941de1755b388cb168cff4df671",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "fc2e1674dab4d9407e5585f2a7caf10a4fc536aa27ddc24116eae8bcf904c636",
    "sha256" + debug_suffix: "759e79a831b8e5ff61e4232023ff613f9e8426b0ff59de66dedce2de89838318",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "eed172a41a05bbd6175024caaa042110adb76f874fcd4b3cb4bcbf70fa0746e8",
    "sha256" + debug_suffix: "8ddd8901c4325203a6f937ff3ce92157e254a009321a718c43223f6f9cc9d968",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "844af7632acdf0626652f53495cc35baf53d91350843f28acaa4ef8afd36d308",
    "sha256" + debug_suffix: "75ac35abaf9fcffc94c41c2b2d99f8ebced4d5f1e345fa8509bb4371dc2c9457",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "8730b1791ff7196f7d61cad9c75495954d235ca59007c42bf06a0dcb92ac539e",
    "sha256" + debug_suffix: "330ea186c6c4dbd17a5bda0b6acaf789dae54dc8f6ba38db9987c42b97dbd70d",
  ],
  "kernels_optimized": [
    "sha256": "ad93dcc90af7d2f4ee089d0ce69bb2d82c230970d8b83520cb8c383a86bb03c1",
    "sha256" + debug_suffix: "9b70c26559f04c99a9f5da2cf8613d7df78600a17b628b0f44bf224578ec9f1b",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "35a44f4b179cef49735b7603d8459fdb6a0a2b2eefdf1ba380614a1917108f8b",
    "sha256" + debug_suffix: "f1df5115f67ffec303b6928059ef4235a6a9c97e29fa308f868020fb60a05892",
  ],
  "kernels_torchao": [
    "sha256": "01031b332e7609303a4f6d3a96b8aa2674064e092fc40d360b76103807386b5b",
    "sha256" + debug_suffix: "3dc40b8a944c1b583a6e6121ef1a9dd125687e2d0c3de6a210597491cd6b3be8",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "0a3997aadcc517b52c74fb2295202160b3b7d131f4b0a069cb6795348e230341",
    "sha256" + debug_suffix: "ad4914f3fa8fee8070535ccca167c4b1a24f450b5112aa0328be94c6ed266b80",
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
