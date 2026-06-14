// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260614"
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
    "sha256": "a36327a950c0bc8f8e86cab06ef2bfb9acbdcb9adf79710fddbfb05161d054b7",
    "sha256" + debug_suffix: "018ed5ff090a2366938ddee6823e54c135f04ac29f77711339680213dd10c58d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "daa3da1c098ac7ad322b34ec75c5c0a28f9c5d74b9c9ba38aac432b43abc66ac",
    "sha256" + debug_suffix: "fe0404d815b8c2e0956e5d84a90b757a1383b129a43f218e21fc1659c641545a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "faddec9af427ca05658eb78baf4340875802b0d62878d008b2fb844cfc503660",
    "sha256" + debug_suffix: "e7936efef8f4494e1d47dae448497be58aba340930e46f5a2733e247bc4e6d76",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "55658e68b40bd811cda9bd73c93c0ee31629c47141cd065abadb0f9f565f9eed",
    "sha256" + debug_suffix: "744d849e75cc706fc7e2b709548f607d8e41077c745a0d06564c57796927a586",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "7f8eaa9bbaa743d3fbe8a336ff3e04b2defc1c88b6a84933a1c79b6568f7ef56",
    "sha256" + debug_suffix: "f64a68e5654f73b97da5c0afcf837f68da30db9ada6d94f0680240cc72018d33",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "f80d908e572efaa4082b7d05fb5e64f1b4e6e3dee8f7dd2cd079abbe41da9350",
    "sha256" + debug_suffix: "bc511ef16afc86fcde8b9c46ef26ee6a721468c12ebe7304484b7572e7e92b50",
  ],
  "kernels_optimized": [
    "sha256": "c674b47c2c4faeb50f0a1e0e8e0f095945df5bedac2bcc6dd98591f4c4bfe863",
    "sha256" + debug_suffix: "e6f3e0c092278820f3b23a72460375567590a062f529f7fd7d41bb82de8cabc5",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a09d12898b321f5320d94e84bb15fc5ad3cff248de6fd5922d15b894c0de239d",
    "sha256" + debug_suffix: "01f0f9fa56cf7a0995cd7b0d84e69ecf0a24423f847f5ebd763224f515adf33b",
  ],
  "kernels_torchao": [
    "sha256": "b25f8c27923eb514b93371c82231eddaea2df6a07a33b3e3f6fc98ea6fbfc5e8",
    "sha256" + debug_suffix: "9754754d3420fc06cc527ccf0c4be6f01ec4098aa2684772f364c0164c138530",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a5eea7929dabcf187939dc0189a1e86d4801d8ec83bc083b61fa9e1c2e3474bc",
    "sha256" + debug_suffix: "bf92a8c8604b4aabebae460a621be9702e218627abeadb0b719de4a5bf875d76",
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
