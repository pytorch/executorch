// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250726"
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
    "sha256": "3c944937616f13929e692ac5760f766163d21847a525ffb6f3970f446e9fea00",
    "sha256" + debug_suffix: "539186bd966262274d3f82384da6d1a99c32e853ce2474882a5ec7cefc604645",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b6602d4abfaae6e40d8f57e11285f9a8ad61afe559bb08ff8c799e0711a79a75",
    "sha256" + debug_suffix: "6b59f1a9e43635d6a6d4f4a7b7d84d78e21297533e089ea56d00e9f8a9640595",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "93c47c085212ae0ff0e661af9c99889319997a8e5aecf92748813a298d17fbe8",
    "sha256" + debug_suffix: "a84ff129f9ec8c241106ab0d96c737c4c8622287d3fdf593f879b06b347acdc5",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "cd6de147721b2ccb8dda0e5a19182b40dc08d3f4e42fc5c6af579b459740bbfd",
    "sha256" + debug_suffix: "d825d7c40c093374b993f13808bea0d4d900ce0c70e69d8626a43cf9c2d68a92",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "2d8aebb172d03f01df360f3b3c7d7aa01f5f0bbc735b6ae2909b19bede9245ca",
    "sha256" + debug_suffix: "41ff0732ba66df80c66343e013b3ada83af1acc27f6b6ce621986cfdafdf81e0",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "4b9c2b64e3b567df53342145dc84cf5c64f7358beb1d1fb22f1af529ab014c98",
    "sha256" + debug_suffix: "487591c48f14db38e754976e5f36862d6f75ce908c2c8ced5961d59e8b78fc88",
  ],
  "kernels_optimized": [
    "sha256": "2ff3ae1f90886ce28cb84f48241dfe89c87e90b04ed3b7c54c3f2bb2759bf21d",
    "sha256" + debug_suffix: "297307dd3c414b2d1afa5dc132ba5df30fff34e2cd18650addbfc187a709ae29",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "95763e33366f5f03d6e5f8a65869319fb13ea4d61b7b8f66abea17a21c36308f",
    "sha256" + debug_suffix: "52d37daa3e18e72b4d21c28e13ee74885b8dfd8ef938ee86c653a3a6fa2b0342",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "2ebb49fe7e1641b643a5d7af2a723dccc52b234333ef14d1a1be57a32869c765",
    "sha256" + debug_suffix: "c3f640cda8fa038202749e3442acb08364e9174bb99ad218f0b24717e70d9c49",
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
