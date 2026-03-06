// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260306"
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
    "sha256": "52a9225828a7ef782f89950a9a4b33b3dd689ae00152d31d99565a137a7febe5",
    "sha256" + debug_suffix: "6a6c3af6e2d0d604001e8e7c7588df1412fd246c80869c83927a23bc9639bcf2",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b19fd78d2d1515d267dc5eb630eda6658629725eac4b52c4a0f77f5d88158060",
    "sha256" + debug_suffix: "0e1c42b730d9dcf2883c75be9f21f21f6cf73287648cd2fd886e1cdd82df5bd3",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "59e8aa736838fd25b3c270926d29be2580621a3716e2d6ab0d79826bac196994",
    "sha256" + debug_suffix: "dd8620f3ed802d6c6a7288833680edb25fa34425c8c1189ff3d6274d35974b53",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "2f8b39d8851290f0c4a1f8c9a6c0a72357aa7c8c750e8023c078e9101a0200d3",
    "sha256" + debug_suffix: "ce953a56651def0ad9b1fa20bf3d1b105471b93e395cc1d9faa497699a3f3df9",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "12b9173ff062f0797de9f4b2cba87367339be5d1f87d0ca96fe72f12722d2f6b",
    "sha256" + debug_suffix: "dd19291f9aab615a078bb54ee490ee9b60940303f17ba2ca9a52cbcf31d5cf8c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "8d57b66cd8b3f3a9c075a3e8e41fabf0b606849c9e1fc2d92bb3de4435eeb967",
    "sha256" + debug_suffix: "1f79492672cbfc1cf1a393388eb676918b324eb405421e6de7af9b849f50f124",
  ],
  "kernels_optimized": [
    "sha256": "220c674953d3f74dba6b1f0ef9f150f788ad29ead89b749ba3dccdc97fbac06a",
    "sha256" + debug_suffix: "217eea45953d92447e2b57e783a150d36b6740fe7c4ea5c2d9379a0cfe374cb2",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "8a47d2ee5f5b9b9b4fd5da1b6add9485238a9a9f4602f87e5ae1cfe540c99ff2",
    "sha256" + debug_suffix: "4c4e5aa728a66d9db59f97fd4cf484fc6ad97e0b9fb4de1e928dc77b7e9c759d",
  ],
  "kernels_torchao": [
    "sha256": "865a4b6e22a1027cf7d3251074735d72f8079d84e25f2a28412d78cce422b3d6",
    "sha256" + debug_suffix: "9f1acbe6256dc431b17645c4936a1daa8dc35ed38c640b3aa9fd36200074f525",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "c403c8fd48648bf7546b26f2e13b3b7b4f59e331fab6c2df647ed953c579fbb4",
    "sha256" + debug_suffix: "41364e3e7be61abba40564fe820ef6507f207cf8f941e68ff69dd14fd0b0e870",
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
