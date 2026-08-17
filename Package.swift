// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260817"
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
    "sha256": "38848b79549955a6923dc0ecf2313471732d0a2a3771da109a7372a21a9009f3",
    "sha256" + debug_suffix: "7c6cb4a6098c3a54adb2ed78b7115a86def31164a657d4670c81b4743213cbb4",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "3a124c5f41d9bf0637359e365fc24526429f2402c1e7716afc31824d066bed6b",
    "sha256" + debug_suffix: "ed209a41ac27f172412fb644f08c850b0ba6eab1d2dbf216859e3b7ac79aec0e",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8fa28353d7b2996f3f59cd200ee207ffcfb6262f7fcc37471ddcbb7f5b4f85ed",
    "sha256" + debug_suffix: "72dd29197d8930641723b8fc9b5c68bbd359bc6750a7f1bd922518ebadaccc93",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "6a382ad016ce92c2e8d90160f9e3b282160d22287b418e374647b5b6f3a312ca",
    "sha256" + debug_suffix: "9f9cf9db0308ba5a76dc3a1019e637db24301b84425ade117ac9495a10bf2361",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "bfa9e9fefc88a6bd9d70e09d56d3b5f0e1c8e2f6659c2f5d3125a5d35bc04998",
    "sha256" + debug_suffix: "d3cc9053aca7759b9ce438b671a59e8e1969998efd1a5b022bcd10b549b9b98f",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "3d3df6b1705c9d5194a110dc38915200762285a23ee617bb2acb06a5ed738bae",
    "sha256" + debug_suffix: "6a5266ca16faca145f15a730e630dd265f2393e15c2e07d4b9e0f624d508d5db",
  ],
  "kernels_optimized": [
    "sha256": "8dab26fedbfa16b34b9505b96528041657a3b08fe237f491cfaf003ab3e39bac",
    "sha256" + debug_suffix: "3fae8ac45e4f6f999baaa9f3f19cdffe3a2a0ae09c52af67fb6bd6a478ef6087",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ff8ee34ff1d26b4f7031d71191b4a94e65ad21c8cd89b473aa13e5d16a702e18",
    "sha256" + debug_suffix: "156bde376714de8c7c7f326f5ff283f8d417b52038744a6e5548ce4b51bb787a",
  ],
  "kernels_torchao": [
    "sha256": "d438c6213fb752c0d37254eefa45f22ae1ce026e9c3ae21428f977d40faccf02",
    "sha256" + debug_suffix: "9912b1ffe80c526e13c50b7017b73cb4ba1b8115f2339d519b6c2b5512fb2278",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "abd72af699fffb30beefd12d5c25f21ec78932bf2e1ea86b6ceda67e1c8d64f6",
    "sha256" + debug_suffix: "2d6dfe0cb820d0af9cce97b8e2631d4505aadfc09832bcd984bb02cde56c0fd3",
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
