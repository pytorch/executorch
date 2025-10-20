// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251020"
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
    "sha256": "16f3421e1fa471c8f1d0ed48ddc00c1cd9ff61a28e81f0585067b17da46a88c6",
    "sha256" + debug_suffix: "bb0d12f7dd9d646eadd3f1955449158563f99d1968b393a2e492a62db2798a8d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "83bf5017bb660cf13e492b3962bae8196cb5477ab952e092e0c4e7069f020715",
    "sha256" + debug_suffix: "c1cf8afb122c04d70795bf1bda36e17019f0d288d32ce4fa55f8fff4bb189690",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "dc8f492643dd44373845376ddb938ca71afdc1c66614a53c9832294905d2674e",
    "sha256" + debug_suffix: "ca9edd18c75e1ae8679ce799f51bc787d3e5a832e1e9196377b03e5161184c97",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "9c5e0e2f3c2e5814eb172f66f4897b8620390f1d9e262480369aa61f30bad316",
    "sha256" + debug_suffix: "71f8dcbab9585cec652acdcee2a8ef7498edf3c2a979c64ed153ea1fc32aa091",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "4d2473e0de658d2d1a77b3215c2577568dea6573767aa47a55920c512adcebf3",
    "sha256" + debug_suffix: "2966743aa1cdc9f4200e5cc64b425e81e73607e448d5629bb15a5309829749ad",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "3fc67637098e78ba69fb94fa79d3b0899cfb44e61da18da3433545ab653ab8df",
    "sha256" + debug_suffix: "d9daa2a398ad1f7e397e79e7732a139591c5ba7a7f43d365954fbb8e3c2cb049",
  ],
  "kernels_optimized": [
    "sha256": "c290c2d15237ae516396051c0bdda01d3709c54ec38e2e4d0c699184cd743c20",
    "sha256" + debug_suffix: "181e16d08a5539fa23889881fe78da4f0e341a7e5e8f23f826a982686bbce8db",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9101203d98eb9c912c9427895404d222712c844ebd8a97e862dbc90e155cae55",
    "sha256" + debug_suffix: "5ea36ed9fbc390540e1d81f8cbd4f1de848f6dc003de22d76076b3bd7b12afef",
  ],
  "kernels_torchao": [
    "sha256": "3a3e2ac266482e7f3604466deaad05767be970153ef6510b9c45490c21c36de5",
    "sha256" + debug_suffix: "723767f86df20c86d35a35e7367eabcd426427da3dd82b9f6a6a67f9553d54dc",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "9aa874b7490d85a37d6405125bef887de71f4b4947094b3ea050a86482b56cae",
    "sha256" + debug_suffix: "d3b7ff73a246de0353a8844eb91b7578d0238f7bec9534f89fe8d4e347f876e8",
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
