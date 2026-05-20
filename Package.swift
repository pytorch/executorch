// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260520"
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
    "sha256": "81b4419388817442fe2d832035502c0f73917aef237ff4fc93b4b2d88e240021",
    "sha256" + debug_suffix: "e3319f78b20e09a94cf37a7d04f2e2d423de5dab9ccbe6f8584cf0523e6eee5b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5dedb72719c684f215f87bc64dd822dfed9d982fccd8f3cb1eae1583697c326a",
    "sha256" + debug_suffix: "7a298383f262b44aa84288971b42fcae81d5dce2cf5622a5ec60b7fead7866b3",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c0918f1543975eb79817191073322d7bd8103d8d764ba123bafcf40aa102a428",
    "sha256" + debug_suffix: "cbfe402a2629107233f50684c5e383d609d42a1661102a0885ae8759dd429b9e",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "1ee22b98efc56eb0665d1b86adaa7eb0a6bab7edd4c3605ba6a895536e6bac66",
    "sha256" + debug_suffix: "8dae0b009905473dd4555ddadc36d00411d1ed46f1f4a0b58c7f3fbd80b18d53",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "10137659376b314cb90a68a95b5bfef2231a73c4eb7ee37c4b105ed09c19f192",
    "sha256" + debug_suffix: "7f049b7a1b4477d3fc3d9b9f95aa7a1c10cee7ae9b61802ca9c9a6a84cac60b7",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "26ed15eae1ae5bc4e9c942a99d606ecc3cc72a803efd98ee23770e5215ed62a3",
    "sha256" + debug_suffix: "2bd547d6102c541aa979e0d6c391b8f193927075d57a13f1d445f49b3b5eb7ac",
  ],
  "kernels_optimized": [
    "sha256": "3ad32a7adb43332f9eaa096d6c141a81a7ac772ff08105750927bef14fe7ede9",
    "sha256" + debug_suffix: "9883f90733e109134f133d561731a996ffa16a72cb7328046c896ce26ccdc8da",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "917858d7fbef17402eefb7009ab47c70a53e36ec9f00236cd28755c5d5eb4c7b",
    "sha256" + debug_suffix: "8798c1c64b70acc236b697d077e15a8e65b963afc73b8f88ac77b16729dcf33a",
  ],
  "kernels_torchao": [
    "sha256": "2dfe559f1a49970c68eaf0b0938611fc58ba8563d3dba3ce802c0226f68d18b7",
    "sha256" + debug_suffix: "79b7f965c4adddf9da53262bf0f3dd2989a8bb31f80c8ed85aa1dd0b1a7f3325",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "2e96b32fa674e893dfff9b7a5aae9efd8340271cc644f01e777a00fc17cd1f6e",
    "sha256" + debug_suffix: "6879ff1003912c17aabe8190dedd8362fdcc55fdf7fa2c9c13d61c77fc52d599",
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
