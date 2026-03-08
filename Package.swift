// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260308"
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
    "sha256": "f321934f1a08a46f400116bf3f197cfb86b2d59dbab79f28cc2c566b67c91885",
    "sha256" + debug_suffix: "a562cb4d7a8afc37f229eae587d95c3c9ef5ea72b44310eab53e6e8c1827fae6",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b7e0771fb21355b20a1d3ab81439a22b78398afe323c8ed4a45eea65bcc82cac",
    "sha256" + debug_suffix: "7ab82c3857ec25a6f1d2948d0f2c4061b637a94574aa6d71ddc3e60ac266d339",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "cd5ce399e30a5c9dc9836fe7d414a725e6215186e8162962c98c06fb22884605",
    "sha256" + debug_suffix: "6023c5469b639ee5b0fc27c2a2e28bb70efb0d8b680fc618e3fe86e86947ce34",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "39e28d9a4398c1a794f0d3649a0b2db126e62e351653357827332f8b9be827a2",
    "sha256" + debug_suffix: "bef509e111d2595eb771e163adcde04c034f60f6a57865a0c2d9267d09964beb",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "7ec377964abc7f40f237fb655e798b311fc9a15a02a4bd1b1111b1660209c466",
    "sha256" + debug_suffix: "13f13242b99244daa243dadd6b48f404db6d437292bdb61877a8d677bab9c5e6",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "a26bda2ba5ab74f276ecb31d1d28cfcbed78d819a0ac789db26b44dc6b3c1360",
    "sha256" + debug_suffix: "37689ae3da3836b5e8ea612e223ba7426690e2455b068d1748a8fcfc08fd12fb",
  ],
  "kernels_optimized": [
    "sha256": "a279e43788ef3d7fa67282d6704ec8d9e4c08eceb45dd9e58d7c817217a43cce",
    "sha256" + debug_suffix: "848e5e5262c7134f304b22d7a8b4d8134d89cbc14d539681fd54fa6ebd0940a1",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "97358299a1fd6af41d55a19e81e79966ce22733281afa92c27934c660594e28d",
    "sha256" + debug_suffix: "7aa4aa95c8db0287aeb5a596cd340aeca527c18262dca174d22f25d623db0e5d",
  ],
  "kernels_torchao": [
    "sha256": "2b85f955a684b9d2cf373225f1bae91184590eb24e91031113a65e1a365e084f",
    "sha256" + debug_suffix: "2cf429dd9bb98469a39c6a570dbd66d0dc6c4ca8ad9d95d69b6856577cb45caa",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "fafa1ceb56d5ec24d3f6e3a22c6a47a357c1fb3461cb27cd521d3ab98d4a5385",
    "sha256" + debug_suffix: "3983463ea277bb120c01f04b7f3cf8b3c819f817034274881790ab484fc2b2ec",
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
