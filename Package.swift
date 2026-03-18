// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260318"
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
    "sha256": "166760d6ee5ecfa68aeae5e2aba40b71326b88309bfc4683f46a1cf87bf07c2a",
    "sha256" + debug_suffix: "54c62eb36018c1b6d360d7c5a3a215ab14f39d2f495cef196ce8f0318ce6ff28",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "9d6aa57dcfd5c19f2d10c741c7f18cf492efac0537bb8e04b926cf05fd41d12b",
    "sha256" + debug_suffix: "4dc8e670526aa5693bf4ce8c5bc2f5687a3ad2116ae836a9f406c95aba249917",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "495f53b5270909a3cda1b30930b746a56d7cbb8e8765ebb8be8cbe7582e387d8",
    "sha256" + debug_suffix: "7b8f96fa831b30c579f7f302bbfb2e44b91d3248b824e49e5ee763de62fac6c9",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "28425fad4e33d08b76e8d89d3139bfe840b0b135900c41e0a294b2ef67e19155",
    "sha256" + debug_suffix: "15059de966e5238a0d60f8d3819a603f32d8870f554b3731b4365279124c2873",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "6bfd31f2bd5ba184455388be599afc05004a8b6bad29aeb3b20dd0831bd3f3d7",
    "sha256" + debug_suffix: "4b036ac9eee7d555aac62d80c2cbdae2e3f84d1cb5632e96a5646255bde0cfdc",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "da720ab492ae28cef34d453972be5e394810dc517f36b5107acb965b6e7c784c",
    "sha256" + debug_suffix: "0d67b5452c2757ff3031fc2d3fecf73166be6dee52686cbd25bd2d80636f719a",
  ],
  "kernels_optimized": [
    "sha256": "bebefe6faff96e58f2c60ba74f3f83e8babb8867563a1d458f1113aefea9bcab",
    "sha256" + debug_suffix: "7c0d6313828368560abe8126bdafe98ab28f987482916278f25eedcca568e2c1",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "d07bc40c6591defbca191bbd656eee980d48811d8b6675263059baa9aa98c545",
    "sha256" + debug_suffix: "5ca6b84fe19270db2a0631359ca5406757861984bc98c7fd4544eba3f75abfd1",
  ],
  "kernels_torchao": [
    "sha256": "fe1d5a19dbddf162e82302578f0d647fe01119d4f87c4901007c8c99f170c41a",
    "sha256" + debug_suffix: "869ea8c339573e28f85895f991bb0a80eb11c1e125bcdc2003b13d18fb2ec6c5",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e7327ad1545e00f50a783b6caa91211e0c4ce43278b00e1aec3e59f19dd36161",
    "sha256" + debug_suffix: "46b2c3610a2419bc36ea9934e0e92ee3f894093194898f347437ad50b557499e",
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
