// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250623"
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
    "sha256": "d94d36fcf19aecb2a8b011cd4b2d07cf39f1e552a59eeee903d8c88948852cf0",
    "sha256" + debug_suffix: "7b9b05542bfa50f228bcb11bc0c7ef8b23a3c21ae16f1643e6de20eff0dcab36",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "3713cc5faeef92f194b747ff198df6e119fc2938791d7866ade0c3dd574e2e31",
    "sha256" + debug_suffix: "808727c9b89e0f40aec7959cb73b9adae48069489a99741870f80b1fcb80b4d2",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "966411c62d974314102ae7c2a5642d8397ed7070a3e8ea8906ba164737806042",
    "sha256" + debug_suffix: "8d02a9c0a18a19a878d135f9dcbc5cf24d488f0ec782a2e4499eda7003190ce4",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "a783a516d6de979c631cb5da3828028d6f799c5b8b15c99cd8a8222243e10d6f",
    "sha256" + debug_suffix: "49ab789f34ee1e07da1f4575870bb882258446081c7cab0484aae32ac4928462",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "eec91585d86d47bbec82e3cbf38987c9e6231a73d59cac4e343573484425cd03",
    "sha256" + debug_suffix: "e01a371da41e3f7614264216b613949c16fd4c60899a297cb91f036586b126ae",
  ],
  "kernels_optimized": [
    "sha256": "41faf88de63c5a4b8bc78b9508d74f9309921d22ff4c9264b3d1c9ce39556d91",
    "sha256" + debug_suffix: "dedea5db207eeb9fa24fbb72984f6584349aa8a0fea9eb2d30361f7af9308deb",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "94f6975bca053cb4a167039b31226b36b3377985c223167735e9b9aed53ac977",
    "sha256" + debug_suffix: "0dcdfa7f7a0c216fcbfd0fd7d24523dedf6669c34b8ca2679f0895f3c4505330",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "588675c495fe9a22445ed3d4f0205cc5cc5d9ebb76b3b5af05454d93b86db809",
    "sha256" + debug_suffix: "71e5c812896b61fffb9c019d1fcd5ae52f6fcfd7cedc11f0b6d4cfeb31678bd4",
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
