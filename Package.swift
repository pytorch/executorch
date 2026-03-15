// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260315"
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
    "sha256": "bab2c53dbf6b9fe6cd5a04bd85710e6299a2cebea2a911051990224e3f459c7b",
    "sha256" + debug_suffix: "39bac24bbb6571464ce6fb93b402be55f0fdc76873633c54035d854d98ea89fc",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "10fa090c6d32ffaece802cb9d89cbefa1f96afe9ad5b1900423e9ffb1cc52131",
    "sha256" + debug_suffix: "61f3f17a8f6c4ea99f80a87054d94527d2210ff87070ae6c6d0c4a4b31ff8b84",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "eead2b5bc43562ff57f3ce2d70c7c1bdcbcc80d41049bf2a4aa5a3510ab868f0",
    "sha256" + debug_suffix: "6f1bfbdd0384e44e5597a6a0eb0e7343f706bab5e1404a47608a8e6da37f6b1c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "72f645ec2c7d9bbf2c1afa20551204c815184e0d971430a694730415c1169335",
    "sha256" + debug_suffix: "e21f4a98a8ed9db29c980f9068884be546fffbfbf0a7b701591e96678e16231b",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "07ef43f36b4236c68b07797809b0c2eff89dc7e2c019faddd2113947679b5953",
    "sha256" + debug_suffix: "402ff682466a72f9bb2afe383d0fc0ee3a4955f81d440610a0a7bcaefe450cd6",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "dfd2073e51f33f828f1bbc162d2e8fa07c44370edf5ba3327fedad8cbc58ee23",
    "sha256" + debug_suffix: "1c632915fccad084233aecdd0a82fcd66a054100afd88b3dfa459e0bd0468d3f",
  ],
  "kernels_optimized": [
    "sha256": "d0ec8c2a58cbbfbb2f05c1e715e0c3c769de422891e9f223d7e639c25898a8d8",
    "sha256" + debug_suffix: "ec89253b765421d8e465b65f783bc246075238f4f1d7a9a541787921eceb487c",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "702919bd4ccba4f0b19fdadea3f85da4bc6086543153adf1d82c5567dd883b99",
    "sha256" + debug_suffix: "876e7a817752f61904bf261fc9052cc689e0c35d3d63b23dab031f86d5620b1d",
  ],
  "kernels_torchao": [
    "sha256": "7df1ff526f4a4caa9dd951f952ea15d375cfe140e3152e52972ac265e48aabec",
    "sha256" + debug_suffix: "522949bcdbbcd6253f7e2f4c02347286ab5d6f391dc0444147e8951f31f135d8",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "db75d36c5b6d947d1f33420563017612a1aadf023c677374e9542fa28d5f5f6d",
    "sha256" + debug_suffix: "014d33edb70e9f51c07dbd82e7848ef58a69e6994ce4ab58c19cda83424e978b",
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
