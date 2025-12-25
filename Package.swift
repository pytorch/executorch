// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251225"
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
    "sha256": "9d9daf25dec4ae5c95e02a3ceb60075d91e10b7c5614b6ad9d05460768862c3e",
    "sha256" + debug_suffix: "4369b99b44028c8f762064e2cf3bb887c375473c430fc82bdc9e2a39f166221d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2c38b566f7b85aaabed22b60bb84078cd75db5fd6f016485095e13306959f51d",
    "sha256" + debug_suffix: "f4eb1ca8069222710047a67926a175af7a6b262788ba3cc9583c6c22b7587002",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7c259924e0b752d234212593a89b84ec70a4c2b2000747027690e4ac47048c0c",
    "sha256" + debug_suffix: "4c4f5551a9b07963f5631b4058bb223a9bc3fe275c5a8462ef48a3c68c2ef02e",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f9a40859e0d311ed86b12c1040ab4f550e2847808ddedbaa82ee00be09814e91",
    "sha256" + debug_suffix: "bc6991a9a52a00b1570e8d3e549ca81f5d2bce6803b21fe645131aac93c05940",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "d73c40ea05e5971eb3e55d658176485476049c2ecedef058ddc75fa1b2f194bd",
    "sha256" + debug_suffix: "1722f1133e323c6610b8144db1c318859b4ec3cc96ee42cff5d632ce0bd4ced4",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "1fe2fb3264c3bfeba2ed56464102dd93ea259edd61308e8b7f3256cba7adff6a",
    "sha256" + debug_suffix: "b51edb8e9fe7d13491a6e75de09b0df2d732bbcbdd3e685c23c92ed9ead22b11",
  ],
  "kernels_optimized": [
    "sha256": "610d908421a410a59bb36a122cd293714676f01ccc1c4ac27bd84a836bd1ac43",
    "sha256" + debug_suffix: "32d5203ead8266815ffb3015f58cc497b52abf5e7ae33ff59ef5b428f7f4f5b1",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "d83cd8e5c533c952baa63f434de028b6e65ed36192b9129d3953617f82993df8",
    "sha256" + debug_suffix: "c31496fe3e9d9e7fe6d003f7d0660c64fd81fe68bd2f0d9c7f9d0f8abb2c6720",
  ],
  "kernels_torchao": [
    "sha256": "f3d2d401c00a7d5ad24e22e96545aeae3199e602c38cd73cd6f06464629c120a",
    "sha256" + debug_suffix: "2908b06c38ff34c4bd8ff208abccd0699703214187179a5e33bf570631e39755",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "441fcb8666001d513473ab14c7cdb9f928653786061e5dfd1928ad03befb0943",
    "sha256" + debug_suffix: "f0c8550f554ebd4b0d2f17570f63b077b116baf5a1eef6e65d07d22d16910508",
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
