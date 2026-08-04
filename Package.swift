// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260804"
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
    "sha256": "dfddca8c3b4f94b73716f314e423d573ced9c6f3bf7805c20a41495abfa278c7",
    "sha256" + debug_suffix: "df3b23f3dc973055eb541cf637ccd435e3299823f5bcbffa41d3b63311346e16",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "0186d2d9ec94e7106e6909426b066a706835c52149d3fac378abf492262b6bea",
    "sha256" + debug_suffix: "be62aa8cd6f91e5d4db6513332a8e79fe8d8afbec8063b1e4f79acbf0b9d3f95",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ddcf97f38a73c23d8d452da12abb1d9fbd39aa3b746fbe548e39031a511e3b07",
    "sha256" + debug_suffix: "e70c6f04ab81959d600279165277ceb4d028e8f943b43ca0503e9f62c0154ba1",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "57e202471167b11686cb5b185984a1fce689da3f5c2af611811d328f135307ab",
    "sha256" + debug_suffix: "576448158cc23d369ccc607fd0021dd9e45081ecbe754e2bf1d1d59a9df939da",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "8d8abb27e32b08a440c2e3efc838418dd6059859a6aefbae0e2465c051a25dff",
    "sha256" + debug_suffix: "9623ef458717986d0b9bedd2300e30a9c57463b3cc588bf1bdbdcacd47e321eb",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "74d777ab909a5622890820aabaa67265cd857b6cf72227a48bd6180a9f766600",
    "sha256" + debug_suffix: "79a1dd5e67a5416416732dd600882756188fc25eb6e57105c5dcf446c5113cfc",
  ],
  "kernels_optimized": [
    "sha256": "53a35b30316c0be41abf315dd722b73df35639d56140008dfef1e2fca045e3ed",
    "sha256" + debug_suffix: "9237a64fe7e0886cb102c2d180253af62c475dab75ec77cc8a8d91e55078af01",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "52ae48b64de1f459c8051660dff840d05102d33124e07d4bf7add51a11614d4d",
    "sha256" + debug_suffix: "f3bfca5951299405f3cb7cf0237f155d7d1a604314fc855700494c4c6cd940d4",
  ],
  "kernels_torchao": [
    "sha256": "9837fcf8a923c2ff50b68e0bad57d47e039ba8f2d814ded36d207fe18fc08d7b",
    "sha256" + debug_suffix: "d36cb06647ddf86b625381be2aa8b4c674798e07e85ff850a84a8f356f1eeb39",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "2105ffcf106900bd4df1a32c1596442d1890f29cc1ef16f7a232738850db7666",
    "sha256" + debug_suffix: "5a6135c7e70946e99611850486fa7d74361abd16799aa004a7030c6cdf31b441",
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
