// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250815"
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
    "sha256": "65c8964239e78cc5b469f11cef8fe6f3836e8453bc4b1c6fd7d8c4fcf6ef4208",
    "sha256" + debug_suffix: "93e9b8cac4916845bb185c0a19082da4aef8186b1175c2fd8ffbd1b9b8a923ff",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "801833b69a69c675c1c8782ad9fad4e2871eaf8082ca242be7616c719ed2d53b",
    "sha256" + debug_suffix: "9334da56d25d1ed661f6164f00f959a4bc8d8057daaed23c1e6f795c1295af54",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "064d02bde110b5e33adc50948ff181acf74e88bccb65ece46043321cdd1c545e",
    "sha256" + debug_suffix: "cf2519c961054abe82e1f4ec98b1a0101c767cc21365bd1bcced80d5d29a2bf9",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "5c3005448c31b7fea683fc89d732d21d759ee5a0712948563ff92e47e7a2b7e6",
    "sha256" + debug_suffix: "12c773103203f78a077430489682fd29e391be6ddf506b14d286a78834fa5e27",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "536d3aa600d9ef31379cb09c3a6aaa527adde982b09e17e7e74c4895f9375bb8",
    "sha256" + debug_suffix: "299c12c7488742bafd2a959a5c9fe968bf313d8415b5e686e082e59ff7a71d97",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "a414b2916863e3b9594dcdbeab0c017edb148f333d6ee53c810f4818caf2eac5",
    "sha256" + debug_suffix: "ad2a41cf3f71e9b05f6fd4bbcbd7f57f33f1697ae6f5873176ca8ca3ca2b891f",
  ],
  "kernels_optimized": [
    "sha256": "8423222f1a28d5333f310cc6de19ba65e47641f0ef246098634bbf967c52bf8a",
    "sha256" + debug_suffix: "ecd0b2a9adb8de05a1c70491d86598b37eea03966be6b21e807bc8bfc49d6bff",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "fbd8ca434de247910fbcb0f7551eacb4646c432cde2f709d32f68a542b4f83d8",
    "sha256" + debug_suffix: "f3e507c3cfab2b0d71b8240cead3e244675243cc374f73632755abd8a705f97f",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "f8385f1575e5b79575365480eef08004c2092ce94e059606fd1f97c08b3f3207",
    "sha256" + debug_suffix: "84f16bf3dbe5592d6f97509f3f246df9e7e0a1a27bcfd4aa233f4450f7ebe833",
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
