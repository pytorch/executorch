// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250820"
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
    "sha256": "8fb02e6415dc70f1c685c692bede205057cce4c5da1a401e6a1511aa64c5e8b8",
    "sha256" + debug_suffix: "9c465cf6f42402a5d77a9dbbc4a88a0895c87155b067eba3c9a12d100767f179",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "83950796234c0e59672a5e59b2f4b3b765b06839521a7af1dbfa95d70da6a991",
    "sha256" + debug_suffix: "f28a292bcfd85795d0f3b6d640c672340e3a5f3710bd851f2b13b958d3dddedd",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "601b522573862b6ce2219a337888e287eec9be55037f3a8ea6fdff40b8e816be",
    "sha256" + debug_suffix: "18832566062062deac48ff0dcc18a43764418fb668e602444a2d6001d1780cc6",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "9cae496c75ad29efc7fdec25d01b97f0a7dc6c584bcffe8baf82a38f7d194a0b",
    "sha256" + debug_suffix: "9c15e794d19cf2b7f6587db3d799630ec91c4e02307a33b28bb1498d62ac375c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "05d4d8f8c65263282a59e60966cf3683e9bd72b07f3bf94572a87bf656db474b",
    "sha256" + debug_suffix: "04082edf4fafdf6e0f5c20be64afae2e3e099d9db2919d5ba78e2d7f794155cd",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "e18212fbc198d75dc4da1378403da21215a20e8f6997cee1fb9a97e568fe9dce",
    "sha256" + debug_suffix: "b5aee8a134cc5f576e5021d1b5fe5b5a4eb3965ccbd838960263e0aae725221b",
  ],
  "kernels_optimized": [
    "sha256": "70f3547ccadc156ac8bc4db1532cf14f3bd86d7ffd615fcd15c7796029793192",
    "sha256" + debug_suffix: "7bfbb037237b2505f1468eae4e876dbe2cdc730396a7b850906022fa29e2125b",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "d7d006f4cc1be5a8ed2b2e2860f8abbe3f3c281f60a2da200320c9a175e367fb",
    "sha256" + debug_suffix: "e7b97386514f809799d8fb5bdb4c7231cfd8580e5ac19868ea6307d34c96dfde",
  ],
  "kernels_torchao": [
    "sha256": "__SHA256_kernels_torchao__",
    "sha256" + debug_suffix: "__SHA256_kernels_torchao_debug__",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "5c63c404b3f17a9e16d701c4b5bce5d116f6ea70fdd45567009a8887c0aacf2b",
    "sha256" + debug_suffix: "30bde6547e6c0aed9c425642d9f5b5d3b015eb55bac17c91aaa2d708a53c1ebb",
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
