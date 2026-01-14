// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260114"
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
    "sha256": "4a5a10a3aeb23bad6f28f2c849d00e90d1f85b1b5ec4da3bb3cb1b4bac947e3e",
    "sha256" + debug_suffix: "5ab2bf5f6acb942e33a5c139c20f526b169d710dbcd844c230c56005ed4804f0",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "081f73086cb86b8ec3215f5ed7a77ef71c6eb2f9cc7b414b0f8e5421eb7c2aa9",
    "sha256" + debug_suffix: "7835e26ba2e65ba379900c8bf57106452b636f61bc1eebdc228dc373bd0e5974",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "2eea32351c1878295dc329b12be67b8998c28cc8a9eda7a7fe6f95bcd6be1e7c",
    "sha256" + debug_suffix: "e7235b245249b73f747dd858f71b8c24a71741811e459f90fb0787f14cf3683d",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "c4eeb85acfedc634679678c37bbc6c73ed915b1af7aec04b39d8f7bf808a35e9",
    "sha256" + debug_suffix: "044fe9249a17317399758fb0a36fab3c20e7d50e0647ffa0f70ee29dddd64d86",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "811c4eaab554883258b38c7aa038a89aa16549920c972dffafbc6f5b57d501db",
    "sha256" + debug_suffix: "9094892359aa00ba6c9560135a0afefa72db638a556e04d90d83f7571c821d0c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "36573dcee893e7f944f3e8d1acf96189f1f71c3866a508fccde0779dbf3268c5",
    "sha256" + debug_suffix: "5c5059f850ec6e62a3735bd415e7666e470c6d3be0ec82c7c81daf3d19c11bd6",
  ],
  "kernels_optimized": [
    "sha256": "201556c2a9a7475dc4f1cd8a5d21e96bc2c168af56c3c927b84b84899d4df05e",
    "sha256" + debug_suffix: "4554bf7f3b62ba0ec817afddddc912f6a91d5d85c9ccd6d9e84874c55dc9eb27",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "7bc7b6bec53d6cddf688c01e6fb5ed219317d65fa76066c7b98f662f70239c78",
    "sha256" + debug_suffix: "74ca7178a4352b0c34b55877fcc6171d2e2960359477eb8693e41e5f86e435f7",
  ],
  "kernels_torchao": [
    "sha256": "33f85e20ec432f6e0cc953d53cae5e6258473348afcbad46c76853cb9cf985e8",
    "sha256" + debug_suffix: "cd481156a360ff5eb32eb660d9e4d6d18b8521a08cefd35946267fa1f2d54684",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "576e4c2d95b9c1d71bbcc1b8e65397a8c290caa945256a14a6ed67954b0e943b",
    "sha256" + debug_suffix: "b7c722454d234a58fc34a134b58203ddfcd09830a80f9e9844e965fe8e901a0f",
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
