// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250730"
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
    "sha256": "2778918f1b5b45edfc0981540771fed4f52f94912c062b818e520010f2a0e49e",
    "sha256" + debug_suffix: "c0e6f56243952682ccefac52c27b6c5839192f4a818cb7400e30d11fbb5e56d4",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a60a3caeef1bf8207c64706e2775762f68c412f501e2df4ba678daa463814d4a",
    "sha256" + debug_suffix: "d198d533b408c5bfd979cd926c30eacbd6e5272292b70938cdd0863f76abf09c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e0688b01a46bf8b16f4b1988310c8ccb761cc7e808dc72b8ed4f01d02a3e848f",
    "sha256" + debug_suffix: "83c8e678c295c012dac607f0465b6a88ab16a61a6a9c2541dc7014c801d8bbbe",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d9429ecb6f97b3bb69a396997dc740baf5906cc9112ca4220a418bbd2013d598",
    "sha256" + debug_suffix: "c0d8e35695dcceb1e080a0e9e0fd9b882b4fb86d7b887f534dd49d89ec35945d",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "2468fae92c17f5b2747565ee2ccd699fe4a65043605f5f55d10e36239812e973",
    "sha256" + debug_suffix: "529670f07e64f70fb23fe005ee95b11db908b67668da21dd6c8c442523c956e9",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "fb3e428cf17a2f72496dbd4c6733ab9caf7e8dc3f45c8632631c7d0a8655d025",
    "sha256" + debug_suffix: "79a05909bf26e41887dab0b0ef64dc29d65da019fbf2c0fe5e0a45170b763240",
  ],
  "kernels_optimized": [
    "sha256": "61c30829570b791e8981ae5b274e0ca6c6864122663b55a1d1f3970edd03bee7",
    "sha256" + debug_suffix: "0ddf8f1dc447223ee8765770cf78f6fe23a2be5be91db375e720e6f9f2b79653",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "e8b1ea849c65803a505bc8c072418b6cfbd27a50c96de4863acccb170ba60f9a",
    "sha256" + debug_suffix: "a3426888e02d4ef83f11a8c391d50c96dc874e6f9f00bf988358720ea3f8550c",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "1082a5f567d833e608c318aed643088b446fd18f13e219de3ffee6b1a1c67291",
    "sha256" + debug_suffix: "f3b65fe570e2d189c34fb65722a0aaa877d982ff58c1377bd009271cc694ed55",
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
