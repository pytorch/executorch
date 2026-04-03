// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260403"
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
    "sha256": "d82f611d46bb194d5191fd026f8df7496c3db1dbbbba71be2261753d37d82b0b",
    "sha256" + debug_suffix: "1d101113ec9741670f82554903cb52a3d4b873b3b6a2d6eff67aa4dabf43ba71",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5400f8970b20c15db73844824f69bec4e66b4dea36662fc2fde9ad264c4f0e2a",
    "sha256" + debug_suffix: "ea10308eabaeeafa8450f1b4340d4a2716865504c8f80fa92e4f5715e46f459d",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e6472f6c46687e013077a54eb2cb1bdbac228d41c2a5298c3e3d3f3287a5766f",
    "sha256" + debug_suffix: "cbefa4b454d64869517bb4b7b6d68dff2c51e371bb3522070b3c1b37d2d33476",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "1a2d0f409af5959cbeae8e09f3ac1ae1472227bd8199fd0b4984dbbff44d4d8c",
    "sha256" + debug_suffix: "7d278664beee08b016d1613cdf1850dd93ebca541b325c8b6c82fab9b5b0c536",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "f98ba674899cfda94055a997a00b85bcbb268f1d33e0812893d432eb7d3fe169",
    "sha256" + debug_suffix: "1c2c5c8d882880c98a3040ad68f28a14ff6942255489c1f09b89ead301d1bc14",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "ce256d41f1cee049c2c0a7f0cffa35daccd22a353e38ec5400537513b33b6b32",
    "sha256" + debug_suffix: "c39cc61572b0b85e197172a7ba4cab82b1db62acd0c6a9c3911f0c353f681dfc",
  ],
  "kernels_optimized": [
    "sha256": "db15ad8fc59afb0cbdb01e7a58a9164608899947198b63932cfee9cff7e42359",
    "sha256" + debug_suffix: "9c3374d46ef596641f0377330c19d396278d2de730680686a4434d560bf3d6b3",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "fe4cda108de5165434198e812fbe336447e9ce2fd4b57d31f22eb21c537bf426",
    "sha256" + debug_suffix: "efb666cba31a6878b73e2b743ff38dfa9d31fe5398466a643d3108e91c502a49",
  ],
  "kernels_torchao": [
    "sha256": "f47b079d05050a17e51a2fea2adc4e819b1abe894322fdbcdceee07fa4cdcbe0",
    "sha256" + debug_suffix: "b4dde31270b2e509f6b87c07212babe66a5d18b1e6e163f0253bc1d74ef9055b",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "7ecf3327bd3007c3eeaf9e9b13f7eb2cc070087ba5e99e8384194da146658829",
    "sha256" + debug_suffix: "c461d5c3baa2295bb13f73840538079b41031922d9b5ef8eae6b3252ee28b137",
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
