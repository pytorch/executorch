// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260210"
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
    "sha256": "5f242b0fb406672e51cc68276399f645a4db2012a0c7453921bb35fe77d0de38",
    "sha256" + debug_suffix: "0b01eec45659e4e9ca7e406040c732604e917dc7571779cccfbdd9f461fcb4aa",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8cdc8e5f6d5d69894e7add4d49228596b15d0c2a757b565eed06c8aa230c87ee",
    "sha256" + debug_suffix: "a87bd623965394bd81e76465dbacd73399fd3bc975900870ff0db26af7413e3a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c29925c6cb8fa1a32b2672889950e9ff19673d186d6f636d480f5af86ceb7c00",
    "sha256" + debug_suffix: "898df633b3bd6b2fb302ff509ff3681c5e67dfad63979ce7a418e87f6b5ed994",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "4adfa5e3a29b7ca61820589462786d0b6662e90ac8f5cd6ee1ae91adc0dc63ee",
    "sha256" + debug_suffix: "e65ecf5cfb363e4eca13b8c7ae635d07c5bf2082bc0ec486838d13609884206a",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "31441e2770a6b1bee6ca33155be0ae5871f0e6c15bf42f92e0c99a90ec3c7e76",
    "sha256" + debug_suffix: "7031cd284f0f9d44afee1f04031da917b693974ea44db8527a91d43a4df1e64e",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "cbe40fbe59b39fdad03a41f0a7766efcd59ba73b705afdc7c160871461792e58",
    "sha256" + debug_suffix: "e4dc1bc6281d38143eb1a6a3338df01e6b5cd69949f52813af63c628cf257200",
  ],
  "kernels_optimized": [
    "sha256": "b8311eed2bd240ce3084f28c25bcb3d66a8f5269b7ab034c0b8db9f5ff47e818",
    "sha256" + debug_suffix: "ae3ca69c5ecdc9ec41702d2330869d60c46c700f1d2dc8707876f4cf3e0c5c1a",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9d84739ff12739dbd622b4571e1b8f55f0c31b40580676f5598d3fae29a70581",
    "sha256" + debug_suffix: "f91029dcc51563a98ed645e2e422c040c6fc8252cb0df61d4048f632d4fb15af",
  ],
  "kernels_torchao": [
    "sha256": "e8ecd072e6464d396baa0fb96c8e28159de65a1a0607a7233dc037d0ee3c6050",
    "sha256" + debug_suffix: "b2a31eda8ed024897859f6ffa47601da83475b42f2eed5e46c225f5b1caa6005",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "6bcb7b6134a2be483f491d7cd430b09661d3e90b69f218d30d6c909fc3818c0e",
    "sha256" + debug_suffix: "74a86ec1b27c5489326f821bbef8af3ae8c4f4c71bcf5f3f953247f6cb980d85",
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
