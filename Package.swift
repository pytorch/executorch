// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250717"
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
    "sha256": "f4bd3bd6cb2a8af985f5b2f5c2d1f5dbd85b985ae9a4da54949825d07d05df23",
    "sha256" + debug_suffix: "427a698966b4604b9aec4445d7a3359aaf2446aa23ae4c5fe040de2511f9b838",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "3a1c97eaef54690d7dfcab5caa896fe15464443402bfc7499d810b4141f25768",
    "sha256" + debug_suffix: "28c06eae83275082ddf5f3a195d36cc7f3014cfdc1d9b6cb667d550ca4f05549",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c838c2e11e363273965db7279e47245a36c83af1139c49f5c638d337706bfa1d",
    "sha256" + debug_suffix: "d166cef53f7fe97ab1ce569ea91ae72aff2bf79f4a1a8afbb20e48656330bb9c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d2f4aaf7560ad223ef1cb4b1cf3344344ad2594120c2d65fbe21b49969a57df3",
    "sha256" + debug_suffix: "f5570186d5f7381317017f94834bb6447cd003860db065632ff4f10ef2935ea1",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_llm": [
    "sha256": "9c394ebd4a7413fa260d08ae1d07074ef534775d45f4bd52b16f23e9d8dd9691",
    "sha256" + debug_suffix: "c7c4f6f268d50e4262d654dc8347b1203ddc53e566eb7d90802aeb284de4746b",
  ],
  "kernels_optimized": [
    "sha256": "f4f108540433d292c00f009c207aff9e7c53b930721ff8b8b979a9cb089f50ff",
    "sha256" + debug_suffix: "3af8e20f9bc764080a3e0feec4a93bb49b9111445399ab207b2560048593dd3a",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "191ecfd344ee8a7a7cd0f898032df33e8d31f2ddb0dedcba10d66aff57a63797",
    "sha256" + debug_suffix: "91e589e22067e3d38ffe900d4f03189030527a724fb6693160a95063b841fc8c",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "12fabcee0e1b784dbc9ed31589e0747433285c298abb12614dfb4b9e585318a3",
    "sha256" + debug_suffix: "d43f1165696e9cabfccb6f0a187cf23b01fc79e2903c2fe40f8c8185cbd97c0d",
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
