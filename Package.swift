// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250628"
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
    "sha256": "ad35d3b2570159c3f3e408938e5a81d7ed17ce98e779f7a97205523d78c25ff2",
    "sha256" + debug_suffix: "c06d5045e43675a14fd24cb64b46d67ae21a51900eccae7b5ff9221bd2238397",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "fc3f98fa95a64d6d6fdbb874092d3e1758321ac0156a8793c9b9f437b714f717",
    "sha256" + debug_suffix: "ef09cecc3a9b90f38ea0b3be3c18a8dfdc2e7898793dd5469c23f88523042636",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "48283e8704ef9c8d6d08d346e9a1b4d8b1d657a195589828043a72660b317e89",
    "sha256" + debug_suffix: "9bb41e10b7af5113759d92d3d15d5f85150457e70d9d514206605932ef90fb58",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "073fd130e6f9c5e32f7c97811eb9b322433bd80dfe47c16fdbd32f10d7fcd276",
    "sha256" + debug_suffix: "1c87fdd26a8e4be68440c56e135d279d6b1b1e0c9657a805b91bfc6fcbe0aca5",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "a08aeecce5f8ba7f0f28fda20b2fb7c15dc96e87c5c358002d91a7e52fafef6c",
    "sha256" + debug_suffix: "42a04066d72a590e74ad7ac81d0d010a1a762b1f09f9faf6e7c89c95e44b5569",
  ],
  "kernels_optimized": [
    "sha256": "f9e5644340d2003b7cc32b4d9bbcaa87b7df21a3281e83428db3ab022749d501",
    "sha256" + debug_suffix: "03d18bcd18ab055112a00ffe43540733ba36ba38697de94fbd5f03b0f6674bd6",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "14376bb012a6d9e79b34392ddd304edaec5534346530af8044fff9eb050b4189",
    "sha256" + debug_suffix: "4c34763bd477a181374f9ee2efc290dbc7291bd67833b26e4f5f2f4d4c870858",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "30fb59863804ff56d24de1c79e54cfa3fb4d09f839336f524ccd5a9d563d04fb",
    "sha256" + debug_suffix: "db78f421434a0efc90c8f2c5d4b853404dabe081dbccf6dec84b6516b0ab99d5",
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
