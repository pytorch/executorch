// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250915"
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
    "sha256": "890d95b2e7d78ebaa2929870f0b883e69e0c675861a38f13eb2a4a368d2d9845",
    "sha256" + debug_suffix: "f8dd734e369f877912c5d47206c21deae2ef0a7d242f738b61b7aebd7b69421e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d2793a50ff387ef84d2bbe8f9b627b5a831f4c9f9ee10e463aa35eba06b908a2",
    "sha256" + debug_suffix: "a32c265f34fbb5f20819a3ed9e143723ac0d657b2a501bf8084dfcff353dc543",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6108565fce32422ec3612c236a2289644272180f2147d805e8dc7514057afdac",
    "sha256" + debug_suffix: "e9c844433bdbc005d38a25679c026189b4aedd22ec21be536dbd2e0274689f63",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "5defc2ca6418fd28d31a387b4dceca3d19626039e4edace4ecd7c4c8ad325b08",
    "sha256" + debug_suffix: "23031642d26c9d0d267624b2f4fc06cf04b652f780c6f142866a1ea79d471237",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "01c41eab5d0364058079f2ce447a75b06da3339646e8b566273808366ea40808",
    "sha256" + debug_suffix: "d3c168dece5555727504109c7a5734106a3fc08346f26682ca13ad44ee403203",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "de90a470d05959238061da1843b9a02f489d64e37e2a5b780b5be16ec557a61f",
    "sha256" + debug_suffix: "fb503cf6ec6ed2b7d70c081d90c80c385114b6fd2f1c43b733249a68045b98c1",
  ],
  "kernels_optimized": [
    "sha256": "23560d35af87645dbdba4a807152029ebcd939aae434aac4f5c097c5a6aeb2b0",
    "sha256" + debug_suffix: "4f415791d4a744e1c52349862400ff5c7d5e8e01539db6815a98d9ee7a76d2e3",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "f3d00e252fb586a2953fda285be9b23b38cac87d2d94821f9f853e29b5d0ba33",
    "sha256" + debug_suffix: "69432113bb3ae56c0ba377273cca210fb21010f11cb562e36b19c79332e6742f",
  ],
  "kernels_torchao": [
    "sha256": "e3503fa931e4a0683d9691b4af2397100f97d3ff69fae80f08b5a417eaa08a69",
    "sha256" + debug_suffix: "798a17009c0a29ebfb58bc2dfb9650919a94afbdc1a0536fa172fc29f29fd004",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "6640f4a3e380f23a5f250d19949fafdc44cc7063db6aabb80a205e9f857d59b0",
    "sha256" + debug_suffix: "ae534ef37aa18ff81f02175fb844c5bb4109ce567592f65be7f49be44a71ecdd",
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
