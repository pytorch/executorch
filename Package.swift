// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250707"
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
    "sha256": "a76b8fe6ebdd52a35ee14d8c72062e4e3a52eb2be55cd9ad2343e26dbe984125",
    "sha256" + debug_suffix: "e7fada764c3b9ac4babd215ebbaf43c21757655298314ea3b892ed6de4aff9b5",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "e046b914ff423dd544d829a99da58933e0e00f7edc576acc11d5a94b76b7f204",
    "sha256" + debug_suffix: "513ffe8f959971347594be1613cc1e3b7fe4150614ff7ab6c6c4c9b7dc4d458f",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a85b3d0c90e07ffd63bcdfc700413f5a9819696892dff17a7dea8df3663834dc",
    "sha256" + debug_suffix: "a6d12b03bafd56a5a17b1708483a0a67545210113b8476ffe573a822f0d98fb9",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "9f5c834373a750d98fe8f34d96a4ef191846eb2fb9231691ff940291a64d2dc8",
    "sha256" + debug_suffix: "5398209071b8440a2218b86f38bb32b879d802a1c7a8a438a1f95dc56ef04c32",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "ccaf01bb6756b353a7bc68d4fc79f6927060d9caa10a191d4ebb8540d0a12a9e",
    "sha256" + debug_suffix: "72081bab5009977709357bc74c738b78a16d7f2bbc7e533537c72e292f71b3dd",
  ],
  "kernels_optimized": [
    "sha256": "1a4fe304286b361b4844eb6e4e499b36a5ede2353720c29100e57931a01d1c70",
    "sha256" + debug_suffix: "4397ccbb5ab9717dc5b64a76f463c6d92be1abd926adfb2fe949b57a447ec2b4",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "74079fb9e1fe2412c2634b7ca86cd62eefed62008c0762474d08283ab281d66b",
    "sha256" + debug_suffix: "89200261272729acd027e2e5c19ebcdbd9e7f9c4ae5cab9f21928234097cc535",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "340ee1621bb2912538d3da10b2f1be9f6d6c7273fa7fa956ad7a717dbc12f071",
    "sha256" + debug_suffix: "44d2784170bc5d5f3d32555283792b98f692cc53806d668a02a52ac656e44509",
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
