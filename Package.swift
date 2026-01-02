// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260102"
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
    "sha256": "6e31b5690fc13501f748f44e28201c6419c10cbfc873f87a505ea2091c783886",
    "sha256" + debug_suffix: "23ed56a820f0182a7d7e1465966a40e0fac13a661370b8a1d586012b07c259bb",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7b41071d4a495329a732b65af20993cb7eabc7613c2e739a6fd14c130edceb22",
    "sha256" + debug_suffix: "deb9bbf44d56d463f1a14c32e1ec98e70b04d50641f4cdba3802d9a6c3d050c7",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c7f0ff1e43b69db248a0ca06bb467fc71a64d3f0adac69f2910e38afe7e88f0e",
    "sha256" + debug_suffix: "880cab8573e539ac7d3f69df977e7457333b6bc5a23a052b33fa89f6d5a861c7",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "57e31582ffd2764cc97c46e0992d7c3695dffe088cb8c18f1a3c41826d45c947",
    "sha256" + debug_suffix: "76a87cf6a2882a53d23c840eea38fdc9ad936b35eb96e9ac81fe49af2844d232",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "69e67c415fd571b70c246cd3481cc455ae05f6bac082ee690a0ab071f757d89f",
    "sha256" + debug_suffix: "4a6195ea6fc03cd7c9a37afe10669743578b849e84acd5e5eb0d2f2d2a1bab5b",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "a9dae3ef3bf7cd77505d95c28c76a8167766aa12f8577e599fe7c30bd97dd7d4",
    "sha256" + debug_suffix: "f2685df821208223ba24698b3069eb935da03c99c2c1bafb4ba8d96cbf0304e8",
  ],
  "kernels_optimized": [
    "sha256": "86d7c37015a7d595b21672d95bebddecb114ff95dc26a1704b83c2e06fca7144",
    "sha256" + debug_suffix: "3f6843ce9d4a700b83c169359c1f6038495d2c576883f130f549696a2bf8124c",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ba996660e81cfaaadc5137bdf33c6dd8b00e6b1e052a810f0985468ab8d29562",
    "sha256" + debug_suffix: "b6c0fd4c3f2b88e415a80bad91b361332f1649e6904ac202daf9b8db694ba008",
  ],
  "kernels_torchao": [
    "sha256": "ce18423545e76be63ae9577b25880b38d3e6c4e9e45c0e02072a2ff817f44dac",
    "sha256" + debug_suffix: "7696b7d532f3e25bf5a420c478b2dd3d9a3a8bae921be418211876d70f8e7892",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "41d1b7b7e5528dea8d4d673c6cbe55f307eb00a9e021f992a8c318ac0898115d",
    "sha256" + debug_suffix: "958043d98158fc962200ce85609604970007d3d287c189f3b5196c68bb21895b",
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
