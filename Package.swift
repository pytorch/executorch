// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251122"
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
    "sha256": "cdf1c38d64afa34bb0bae311190f2e75503ea5eea2c025aed4bf8602710d7c42",
    "sha256" + debug_suffix: "3f002b1eb8c8746c41f18f4ec665188a7df60b8f0be8e860f6a8818e2c112e43",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "eac49e3c21fdc22884991167fdb8f167b609dabe05b895c97fa43cc4e41b2a04",
    "sha256" + debug_suffix: "26957cfc78418babf7c8a1a473e33847ffee306cb8b834fcd449125bb572b55a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "741d781bf24c30d59a0b38081be80c0cd6b4380d27ce48c5cc858e45eb6d01c6",
    "sha256" + debug_suffix: "7f3ec2a9ab13dcd27a6d9b175b66508b9f50a7ff7c049bd5fab836650c30c684",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "947c977b0d2469abf7c8e2965629cd7fcf2a4fe961affee79e64f6909bb6ed25",
    "sha256" + debug_suffix: "04d76ad4ebcfb4cec7ffa3a1e7fa3a4f5dd4dea75357b028139cf8f3607b764d",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "272dabfe852e619edec3963fa241e61db16e1da54d793937ad6ff97b55117d14",
    "sha256" + debug_suffix: "02224a160671128e400c74e02ad941275320296693936095e4271bf231cddcda",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "0b6f41c395534cd611cfd26c859550c94aa37a030c56ce3be6d5bb02d8a4fab6",
    "sha256" + debug_suffix: "f6186ee1e9222aa8e3c0b1c763343a0a1ef1e38a54cb2a26ff35b758f33b5998",
  ],
  "kernels_optimized": [
    "sha256": "ee591e9baa874c67bfa565a2cc459934470da1c63c825be726219bd4b247112e",
    "sha256" + debug_suffix: "e71744e17b63b519468820e76e0527b19337206c80a8559805057366141811ff",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "77fcfe202707cccd7044c65461ef2486857477331095b4e7a81c28024521162d",
    "sha256" + debug_suffix: "66391decaeba7fd7dce1682ab76a092b3506033e7a71e9e0cb4a128bcd381126",
  ],
  "kernels_torchao": [
    "sha256": "9b02b12a92e07fcf512706ec0170d3870a2cadb50d242776c2cf036139cfbb35",
    "sha256" + debug_suffix: "00d27738296b35f1ff9c6983c7b85de075588233f3611f5573b25f766f2940ba",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "d5ca4cb1e82d619b17d3f672f2c79512b7da42619184b78c001202c92e61b951",
    "sha256" + debug_suffix: "7aa9ea883f05af03cb2d852a60c90151ba0711ffb848dcb2357dd118d0f4774d",
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
