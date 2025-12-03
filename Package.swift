// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251203"
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
    "sha256": "969bb12a5790d09d209b166d4d5d49bd16d8474bbbd6dffca0e47886c6d55f6c",
    "sha256" + debug_suffix: "7c644038fc1a7c7aeb731402cf873f19f777bb959c5630e6d12cceebcbcf7993",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "05b364502f8a522bc0d5364a1a4985432e42d0d7577891f52da4a6dacf120384",
    "sha256" + debug_suffix: "51e341803a1a9bd48932b1be8a8dda2d1c28b291baa9f2d956507eab097a1f38",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f070c0d89d3523a884707387bd7469aac5276a463c80f7dae9c3c3a31ddcf295",
    "sha256" + debug_suffix: "ed66dd00083073433502ebab0da92d5047738d4d60fdc1683d3a4e5edea1c6df",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0dcebbddeff71547a327c495240dc756fe12b6f467a5eb3a01139528368c212f",
    "sha256" + debug_suffix: "62560444fbcfa9a4aa66dba176d976cbc9610b018bacb1d7e2b553c88247b667",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e7d987b125c4f3269a0f5b13f8800211611569b03c202b6a51c7698ef9143fdc",
    "sha256" + debug_suffix: "1d8395041a27f06418b58fa0ca8b18e58de671ed34656f19c42d5c039a7994bf",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "7fee72ba4a07491112689fb732b4d6c8e613c3fc2d78b28c84dc790a22e32d93",
    "sha256" + debug_suffix: "4117f3f68de6b4b7bb7bede860f0a036c2cefbf5f26b29cc303a56dbcb3d970a",
  ],
  "kernels_optimized": [
    "sha256": "d04deb1f44af25b5b86ff5a65cf4087b8658f1060bb4139e3ae1f3c260ed753d",
    "sha256" + debug_suffix: "5a3f5ab1f003de75a79af405da33f74854c840991494124b17e289e56d00159b",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "5a3e0ef6f28c550613266cc22f81a1895fde1e3cac2eecc07c441e692ca27a8e",
    "sha256" + debug_suffix: "3c15bcfc5b692be548d1eba83e87ba84491e39d0787bb848be5cb7e9f5ed007e",
  ],
  "kernels_torchao": [
    "sha256": "792a02b24bfb8d90871a9e3d5c7981447ca7b0fc6a8f8ce170ae62097e1362be",
    "sha256" + debug_suffix: "498da8a5f6f3724c7349454c65892e1e66558d4450a9056e3a4a1591e890185a",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "5d3cc3c81db098a8fe969bb912c4639a589a09d20827c74f99f58ea623ba63e8",
    "sha256" + debug_suffix: "d03b71b743bffcf7eb73dc8a027c1f27410d6e3f01195d8af6907c1ea4c4a5f9",
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
