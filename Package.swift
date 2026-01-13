// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260113"
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
    "sha256": "afcdba294da48edae131eabdebc7d34749f55f7c03737c1c678e750e9e01e652",
    "sha256" + debug_suffix: "8f81b25debd0d6f538a1e4fcab34706a909bc4338569f13b152baa9506e5d067",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "741243fa99376b54d9f592acad5912feabf494db5b0cd3084cf41436e8e825ed",
    "sha256" + debug_suffix: "26ee485b84a87687817a0c443c0b4a826f2d49a379c6250b526c1535a1bb41ec",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a71fda84c051e23cab44b78fee96a2e9d0bce99713c3cd99f2700b2890217df1",
    "sha256" + debug_suffix: "8e0b02d7097b5ac0788b2060d73ba522ca09a00fec6617d1369a40b41a0ee031",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "61c600a58f3fb0f22a9e459163f41d0dc2ab52a4ae2582123bf764ff7cf39ffe",
    "sha256" + debug_suffix: "49ee938fd168856edcc73990f7e88f06b29a6bcf6764749af877e3a3dd4abf07",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "4eea1a38e4e393f743e669f68a44b66c41a044b3549caaf1c18e588b2a2ef3d8",
    "sha256" + debug_suffix: "a6b8219f4cb62cfc13920a6deffd861b143987013597a57eb18c80bd756574b3",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "2d8a39ee10eba6651bd0da3f1141fa5a762e1a1aeb6761dcbcac665c6a765f95",
    "sha256" + debug_suffix: "4ed38608f9b5b081cfe0274485c7d0c1fcc06ab411e223b30ad67a020f7d854d",
  ],
  "kernels_optimized": [
    "sha256": "76db83a9dac7469306d78a75d98d27531fe6c69141f29de24e60b17069884fc3",
    "sha256" + debug_suffix: "0121be54be84e439517c0bcaa49238dba3296a6396e96f0338ea7be51a6c3392",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "6d5924134c0dda140e6c35f87e123099f0366db206c99bda1640520e65e51b13",
    "sha256" + debug_suffix: "e7ed304d3fe07f086c7165213e3355c840c39d8da48f7ebe619bab7962c421d7",
  ],
  "kernels_torchao": [
    "sha256": "17c4d8e04f9010d18fa44ff18cc80ad35d2009766d741006f54e643b9d4d931d",
    "sha256" + debug_suffix: "9c0eed034073827d3b2804a3fa8e93b6e780bdccb95383420bf1c022d3464e55",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "ca29036560c1d98e70f22fff859e1a48fe4c77614c3ea6121958fb775875787d",
    "sha256" + debug_suffix: "84c8a3b05a9d045c34831f17f3f114c40c3bc0c8dafe3da2d0218e50ff17a57e",
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
