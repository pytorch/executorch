// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251014"
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
    "sha256": "1b14ada054d47ed9ef8ad8b9239186dfa879a104b11d835f241425fcea56e0b5",
    "sha256" + debug_suffix: "15e3779d66bee9f78a93ba71b10ca3c8e9d0ae73449829b098d3e5d4624d9106",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "926b78bef2e6e5ca1dced68d511d95ea98ab87ba34755cda34a051b88d72d7c9",
    "sha256" + debug_suffix: "803a12b5d85490f8e9edbcdc9c3eab53eeaa35802922bc9e018f0bd97f15a344",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "5f0e9cdc10849e74f549ac184f5ec35254000d3c9ab848e10a6682e37e42a2fd",
    "sha256" + debug_suffix: "b7f85dae1d8bfdbd64581a052148670045ec341ca54b71bfd5c5971b3473250d",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "25c2b580c2ecfd4e4fb3c8c7d05359a5f3190dbec72d0a6c65c8124ae84bbfe6",
    "sha256" + debug_suffix: "88dd296f89614729e8cba6a1bf9eed9ff5bf677e2c62d54ed2d193b5c014e850",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "55b60b4f4c879b894c2111536adac31e0503999d0576df1dd3bbfac0f0e81bfc",
    "sha256" + debug_suffix: "07656693e8019e91f4e3736b07a4d5001c9bba249abfb6ea76e655daad220fa1",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "e4ec2ebaedebf7ca558fff7af4708c54000041b07df18c6a582d6881017148ad",
    "sha256" + debug_suffix: "728ca627dac22500804bad8ed1b96f9c70069b88b3df970da70ac6d1e7cc7ded",
  ],
  "kernels_optimized": [
    "sha256": "5391ddb128aa6ecf2cc226e223f1907f27a89a229ada778d6a2fd2a05c5fe7b0",
    "sha256" + debug_suffix: "779cb6000bcead765174e92f0200ed7070a825269d1a52a34fb2e22d0c3d5726",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9ecfb2b8b423ba601aa81ec05ba8bf7abd9746867be685bcd3035b18f1f430e8",
    "sha256" + debug_suffix: "87b916cbcaa90adff265f2d83b05c2f5711c98ece5bb76055317dc21f84cd384",
  ],
  "kernels_torchao": [
    "sha256": "282d947567daf42c23f8b5400b3503bc14128b4e83f7af93bfb00dbb66d8b0e6",
    "sha256" + debug_suffix: "03663f460e9e8b00ad39fbb6c7e9b40c1a2e6c825c7826fbe330791b9800e49b",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "b0e1eee498acff14036d9b9dc3e17ac17a6ea6f1632000db9f3502badee2a3f1",
    "sha256" + debug_suffix: "42fb3ee818c186c7dd0be3026366a5f136cd1a7678c15af1567de3865c1d194f",
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
