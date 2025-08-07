// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250807"
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
    "sha256": "96c36c7d584b1f27b610697dd78f171c545b041acafe153aef7dd8f1ed4f2e25",
    "sha256" + debug_suffix: "e40b3759372775e06725bb59c77833bfa8cc81e80561d31ccef92dd0437d6d4c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7c799620a5930e4d2abb1737c7e3dd452bad2f37208c048e50ffcc26e39231ad",
    "sha256" + debug_suffix: "76f7d2e9ddd18c8ccef5df544e63fe753ca3793f5758ebc178eeb81d7857af02",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6a0d767e4af01109fd4b87e7a4f9831498f5cf17e64e706088d84212fc96bc28",
    "sha256" + debug_suffix: "510104fb5addccf7441751045d1319a0b6782086560999ab6540f730260b2498",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d10f559401153ce512a65c43707a83f2a8aba23d2ca9bfaa796f5e1fa0c3b80d",
    "sha256" + debug_suffix: "66919a2bfbce89a0f5a70e8b9fcbadef911dc9c2189b67add907630168bf074c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "2d8c18f9a4a5f576475bb39bed606334b29bfcd9c514a15a1461a36b2bdea6f0",
    "sha256" + debug_suffix: "b70c51fb7ba5bcfacf8a914eb2fb29729350fc4d19fdedc05640dab35d32ce3b",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "6c93f8221b3198b8f4b3e41a34d030f223c5239f5bb7b5236158f7aba26451f0",
    "sha256" + debug_suffix: "42871afe98548bcf34f5540813bd75af68557c401658feae12b027f9ebb0aad6",
  ],
  "kernels_optimized": [
    "sha256": "20006400efa8f530d44701e465be360d88c6172168b2c11b82667a938673b537",
    "sha256" + debug_suffix: "07b01659089b376377377c2d5b6a675e4d24147dab116f901caead3f54c5a364",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "594d1772439cd25f9f3f15d287699fa3fdab06fa33748613059ffe1de70df00f",
    "sha256" + debug_suffix: "d0c88a3edf511dc37ce174a10b764d5003a3073c2a3d64a07443735a03fed0fb",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "d4802dd75eedfc8dbe80c0ff008a84c77187dd291c9710009bc36b964dd272b8",
    "sha256" + debug_suffix: "c657473320861d14d639e8cced88068d3ef0b9aff826cdab413dca5bd39aebcc",
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
