// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260815"
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
    "sha256": "bb388ddf31f8148632ebc5e48d960bd589bf088c973b71eb1da0441e2f9ac936",
    "sha256" + debug_suffix: "1014f1c75951c13d00597891578e734f1097e2f640a265b7bfd36000b74e8216",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a6a2acdb85742c139954da53e0acd96ff46880204e5900f0e3cc4d1f68eac1d7",
    "sha256" + debug_suffix: "4ffd21635ad3bf101d19f8ccd71d108e70fe00f9ec03e222f9d471205672796c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b41ce38cac060ff6b2207d69ded9e26f5e0f607d54bd55d9916f584a55226d1c",
    "sha256" + debug_suffix: "57154f27b4ebc8c47cad45448fb9fc4d330d4da789dfc0cdc79cc032ab66540f",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "cd238145704fa0a93d0d1f28525e80e1e97a14a5f8bef164a4bffdf69d299856",
    "sha256" + debug_suffix: "4c729f99a75ca3885bf3951fc93561917d7fd52cdd0728fd2b5e1974ae0de3a7",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "c991e2e4f294e1f27a4ef3e99404eee7363d49c30bc25248896ddf4cc6eed966",
    "sha256" + debug_suffix: "288a13d97ce1339a4389d18e3857569d85830464601babc4667769ee9067f624",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "3b306bfe957a963058ae454a034b72d185e91e43b3ceed1e10c6dcc9b6e8e26d",
    "sha256" + debug_suffix: "84bfb63d68b3d5482d7253c4d4e04eac623ed2be6eb4ec89345f6f34db40205d",
  ],
  "kernels_optimized": [
    "sha256": "0b4dcc2e7c4ec6c3d7047546d34d3ad1fe5d41dc315e52a052f1b43d889f28bb",
    "sha256" + debug_suffix: "a33567c9690eb90e9669a97b566917e2b577c75648252f5247a19057a1606e06",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "bdf86e435c7f9d89ab5c22a1e0f12676cfc17b4973a31e2e73661c0b0bf00b81",
    "sha256" + debug_suffix: "4a59ce1ed90b77bb6a36b385f9c42b253ddc4b922b43ee13f6cb64bab990a490",
  ],
  "kernels_torchao": [
    "sha256": "2154af0029f65528ac8685eee3b9ebaa317f6e5b77718d14c44d43ae5d6a2c7a",
    "sha256" + debug_suffix: "98ea8988ed1e723b8dc852833867d509a244f96086a7de4fb325df1c6cb5e371",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "4b97308ac6f71a749e83f0d08e4c0e6155eec23c4c64e7716693e0bd4837290f",
    "sha256" + debug_suffix: "ff14f11320ba6dc8e9406dd388e9d17794c00bca51398a2b47ecfd485750dc5b",
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
