// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260609"
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
    "sha256": "5c80bdcbbeabc31f6a184defd4daaba6da25da743100864cf5e22871e8153458",
    "sha256" + debug_suffix: "943a871ea49d4d684417a530eb09a011de68d36c7d77122c7945f3c37bc8c774",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5e6c87f038ded9f302fecb5c0472c5e7ab5d335329a49fa1bd04c4600b6127aa",
    "sha256" + debug_suffix: "e1e779a78c562d01ca0ded391c0a89a7cedc46dd267d8e4596a8a735b04b0b51",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1fb67c67887c9a7bc28d751c87f7f6c6e68a577a83f8e6bff3b446972a7291f0",
    "sha256" + debug_suffix: "5ca09a3842ce352cb2cbf6fdc44ccf7f9017e8eb0ea987ddd0c0fc3ee8ca55be",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "6b21f08015a522a26d7ed0bd9b0070c4162a154ebba513e2b00f01453b642d33",
    "sha256" + debug_suffix: "ed73afd078a9319408fea2cc1a291fed3d1f5ef881c08f39fa116cdb6b485863",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "a459bbd830bd7e6f16a33ab8b88e60d9a6a798ee7f9f62d6911b2996c4a6afe3",
    "sha256" + debug_suffix: "34a93785f0c7367b65179a4da52b99a95077be78fc0a9de00487f4d053a5e692",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "faf470262fdf4d99a9dcbd230486defaaa0155722bc33a135bac4aab297d7de8",
    "sha256" + debug_suffix: "a2084135a72c838db66fafa8d0bb786ac8d982ab42425f00fb54998032fad66a",
  ],
  "kernels_optimized": [
    "sha256": "dc29a3552aeb7b3f81efa24b791386845d03f68fc4480714a45ad1cdc60776f6",
    "sha256" + debug_suffix: "5ed88634eb23dcee4d0b871cea01dc26f7fb36100bcffa38ce793aefc2c4dc48",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "bb1da25dad45731e22cb30dc5ef5afa9c6818630f68f3ff4633109ad64803e41",
    "sha256" + debug_suffix: "dd013c8439f1a6be333a74ceb1f8e9d6ae747012fb63d4ba321dee87f1d577ce",
  ],
  "kernels_torchao": [
    "sha256": "ef3f7ca87687a96c28da000adc7f4e5dd461b53bee8388dd0c3db6f385f328ea",
    "sha256" + debug_suffix: "789ec42f3ae4035fd50d63ee48dec5a6ad4fd4fa0686e33367c9618298b796f6",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "c2ed492a287ba197a96ec4408268240a78d7f4547f73cd8eac240b73d4280334",
    "sha256" + debug_suffix: "7e368100d6d982d59c1cfef680200e93d89250e39a25cab2abefab0b10a8967a",
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
