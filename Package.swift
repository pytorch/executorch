// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260128"
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
    "sha256": "98a1117e110dbab5ca0f045aabc5244ac77f4e13309d10400b1691cb9105aaaa",
    "sha256" + debug_suffix: "3c7628ddffaa41bb2dd92f3ffe987342fdf535aa741dbaaaa6294012fc7747d2",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a9f8e4201b81cfb47939ad87d76136f7500ced5ceb246709b339bda13a5cc1d6",
    "sha256" + debug_suffix: "b3b4daaea9bf0aa84b9371428e060477a88c50a5c62d33c02fcdd6151758b011",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "de9d0e4eeb3928f573fd6a633254c82025193cd8e20661d28123af9ac329e099",
    "sha256" + debug_suffix: "02f1cc45bf01709544a1aca959cf491ee6a39377bf47cb733d7164ac8723225d",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "9c9a91c9abc8a1540f1202d49c9a7088ea14eae11413c0c2d95009f20ab7dbbc",
    "sha256" + debug_suffix: "ec0889039b93ffc25ef073617b3db0ef2e58ddfe9a38f265779c06fdbd85f660",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "357d8629c6e8ef8f903da02b8c1802fa33aa362cdf88bf637ea3bd8527eafc11",
    "sha256" + debug_suffix: "7e6376d407565041342a0ba6c249a172e51c1caced9413c91d3c195a5c2e202b",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c6331b3cdcc7c0f1569cc6dbdc0c48a23f035e0abc0a40c46c40e7912258d703",
    "sha256" + debug_suffix: "9d641929ead08eae5585725728ba67db90d823b70904b98511dda139af2ee5dc",
  ],
  "kernels_optimized": [
    "sha256": "d20b2cd68998b1a9d8357ff8f276ec27d1b0a69b2a0a02eac1a54459d4bc6259",
    "sha256" + debug_suffix: "ef0caca11121ff3e1dac6235e8d70d54bc2fd4956a070365a4a6ae11799343ad",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b5e1cbffe09ff7567f7153c664229928af2c6a8a6798d5b73091fabd99e61772",
    "sha256" + debug_suffix: "f24c2ee85040d729ac7c520441b5b1fd2c7e51ae1a41ad49d0fc7213d24cda01",
  ],
  "kernels_torchao": [
    "sha256": "0360eee34435370e7d8789d4c5f6cedb1d06b1ef83bac06fea8513d19495d0b5",
    "sha256" + debug_suffix: "161474592c7bab17e52a63fe5c21aa10f7f71dad32515ff173e3d91142230ee7",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "fdee39bc9102c39d679b87030e0401360c11fb03ffdca7f786549d32e0aa6cda",
    "sha256" + debug_suffix: "7fade6d4aa59d9f3583e4815de430ecfc81d0261e6b078d13fe322ab35938b22",
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
