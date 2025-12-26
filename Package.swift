// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251226"
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
    "sha256": "ccfa79d787ea6845df421458cf865f7986b1a5318f9478563689f712521a3544",
    "sha256" + debug_suffix: "9c5a104d7df96718f661e0a148f36c91ef5c588339577164b8aebec031a7236f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "9837a343c8049c0ddd38415f1e3981de728e1e7f3795a26506d17a5cdc74d4c8",
    "sha256" + debug_suffix: "a891372e847569e5d2d7ffb0e380705adafb6f7eb13eff4c327d872c0f0203cf",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "fe47a6b0c9df40b13881407a3f11cd842e8e132729b9df503eb93a5f70636298",
    "sha256" + debug_suffix: "4df610b9d9b75b943e31301ba02c16768443eb62a870f9bd3bee3aa2083bda90",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "6f027ced7abb59278f7788686d8ca091b6483067f3774984a4ab193b43172d69",
    "sha256" + debug_suffix: "623412bf34387a2149bb0133ca368d8bb9e582c086fe270149b0df95d9108a2b",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "3bd7741c59857849d377e28b02b3bde7a443d729f316510abe941681706cafee",
    "sha256" + debug_suffix: "f6958bc34360cc27b4c923a0f028c7d388546e4f88f245b6c2170f62bd187911",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "104268006b233a27a322f54e33c5e49c8790c151d9164ab1aa1282ba5640b647",
    "sha256" + debug_suffix: "77928214fc72bf5912cdc800c9f480b117386bbdbc435fe7a5f22d9606ce5f2b",
  ],
  "kernels_optimized": [
    "sha256": "f73fb4cfe3850be0dc68a6267147b309b9782ad7084fbcf17586dcf97118db95",
    "sha256" + debug_suffix: "75d7876211b46a6d2eb5ad09db77abfa3241ba87097841804b1406213d57d196",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "770cc95e0e462c6bd796950378ac8081becfbe0eaabcb4012c84c09a209eb1b0",
    "sha256" + debug_suffix: "64217d2e101750b849d18c24cc58b83c045befb4e6e4f6f32a7aaa0d3db6704d",
  ],
  "kernels_torchao": [
    "sha256": "e02908c2eaf5af7f8e914106f2ddb5e2a79c28b99669c33fd09695493f4cf8ee",
    "sha256" + debug_suffix: "a99dde6c30261bbe8ef1b1015258e4bcafe657afeb55381a54c44f286d264942",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "4fc9a8cf821e3865f981acc0ee244cb4801ba9256508958612db42cba9857a42",
    "sha256" + debug_suffix: "a4643911a4f3296c7b2a26d3ecf316df7e4fcc0a250a6edef84e822d7992ae20",
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
