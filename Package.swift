// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260807"
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
    "sha256": "1c28e89150b9ed4abeab50896162b967d57f4f671c1b43e1db87db275d1bc66f",
    "sha256" + debug_suffix: "ff71343bf220ebe112613a40482cd0625735e9698267c7760110c47e465899f2",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5ef3a7fdd38455d54661fadb2bc5b6579b43400531ce16f368c774531b82fd71",
    "sha256" + debug_suffix: "1d5d8efc4874240dffd7e03e460a2fce835d93c1ad28996a1bad02c0e86dde8b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8c753b805da51e63834934cf645cc560d51e9664c2be0f2a4870b00770da1328",
    "sha256" + debug_suffix: "b4c5b07fb15bbf3ba41a109835dfc4358b36330fe493c61b07db016907eb4a6a",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "05c7b32c0478b0ae15ac991c3cabe199a7a8e07fdee0b3f9477cbb7b4a609cd5",
    "sha256" + debug_suffix: "03811a02ba474f6dcef344f262b5319865e9509fa0741a252688dcc51e978c4e",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "6db6c01db4f781f08b933a9dc73613cfdd5b92145f9189eb570867b204a1f238",
    "sha256" + debug_suffix: "5306313e8f71e396bffb339d00c4252fc9a61a86cd6e309b0c078d64314459d8",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "f94e0d9f6fbd4bb25022d673902d6bd9b38ccd55ff850009e026113488c59977",
    "sha256" + debug_suffix: "62439a7c4f75911bf431e7c6a3c23a4d91a4284a084284baee3341e6a63b4b06",
  ],
  "kernels_optimized": [
    "sha256": "19a281c15ed1ca4910f827c5cfb3ee6e11aa7d80fe04ca363f00bc7bc2d1a8a1",
    "sha256" + debug_suffix: "481979c8b5af66dd119fdd2c5b967c6fad88ab4e66422a5b7c13e59ad04f6786",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "4a9fce9cfa5dba8fa97ccb8433c142c091ac0bf2ec3608ab6ab569f51d72aaaf",
    "sha256" + debug_suffix: "86e125ad21e483017a01c0d802eafdf2e8893af062b73e4cc82af24f62a14e8b",
  ],
  "kernels_torchao": [
    "sha256": "1c327f628b3d09acee0ccf45751375eb42f9b0cec80d28bc99ff84837f48f424",
    "sha256" + debug_suffix: "62e234d0f149dbc68f5b84558daacfe0a092b4546f58083eb1945a42f44cfcf9",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "330f78f81d9b41cda0d5ee4438dcb1d7415f4c792e4f24f04171bbd4bfed2131",
    "sha256" + debug_suffix: "a2783ef6f3467cd99003967ba9472f1b437602712bbf6f5b4d77d5a74ef06f97",
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
