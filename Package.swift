// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250912"
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
    "sha256": "37b72cb878602aeae5280c601d041612de986f4daa7549736dbdf7764755462d",
    "sha256" + debug_suffix: "d7984b58b8aa543c23004f104e7b4c1d0b31766275d960d9520f03b0e68d0493",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7fab48afdc8312d42235866756445b971b59d5d04d12ee3b5bceaef62c4e4d4f",
    "sha256" + debug_suffix: "e7872c100ceb702c9910e4c9ee6dcf13aaed7650afafcfb452a858612622104c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "0d826398cbff21d92e05c92753addf1f776589a88b232e3de1a3ee1c4e33b3c6",
    "sha256" + debug_suffix: "48d1424d5027a598ff44ece22315fdc3812e4a2a8e9175835645a212935fc41a",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "cec43527345530a491e2a145e5f9ca1a405d602507b7073ce93b506f1c13aaf5",
    "sha256" + debug_suffix: "bddcdcecd8996a897be2aa9f81aa9946b49023306a52affa44f95617dcda223c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "90c3aac4886935b4dbed784f62ff135fb2748c85429143c62e45826171217b87",
    "sha256" + debug_suffix: "bde15468a2ef324758a2a31d7bd2f586df3b82a2febc7e7c7203d24f735ad4a6",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "f87a1870bdccd9f6cf6674e4b0edaace3a37bff9678da1f7720b5559318522bb",
    "sha256" + debug_suffix: "a83add502b004d0e9885206161cc070b2f78fde7c16bb8b89f34e0bb3d440cfe",
  ],
  "kernels_optimized": [
    "sha256": "cd8002aa21930dbb45e77f8a80a36a347db27028b163c51d3770b1d7b076becb",
    "sha256" + debug_suffix: "6260610a992be16bc2667d5b07388a209975523df9655af36aa2532a4409ea94",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "f7f278d7fef04cd9ea649f4316167238a5f9c6cfc66312ce51cfb515ee172582",
    "sha256" + debug_suffix: "b194b376769ec1febd1868ef8fbbc64fd0c0301f4701c957a6df7452e4b87709",
  ],
  "kernels_torchao": [
    "sha256": "e29e2e9713cf8c0d181e20337aed598068ebcd623965674c76741f5674db33df",
    "sha256" + debug_suffix: "16806875fedbea4489cf48851a81d05f10157ff329220e8ccec190d86f04eb3d",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "8eeb8967708fab39ac1b98d8e7b511b18d79b4a3795a604f089b44c085594c3c",
    "sha256" + debug_suffix: "83daaf3d924c406418ac5ea48f04a57f486c611080f45abae2fd1190279b4ba9",
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
