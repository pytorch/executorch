// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250702"
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
    "sha256": "be888ec0c2dbc24224e78f44f8300f13e5f670d9ccfffa6d4c69cb2d8029a667",
    "sha256" + debug_suffix: "eca0a2df99f05cb4249ed518f3e187a74effae8a34218b9c6ae5a0569d9afec6",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "52509fa28e3950ba86245f9f8ec8bd12d3b5107665c25a9e8975b6997c8e16bf",
    "sha256" + debug_suffix: "ba4ec936af82a6228761a52af838b8617b8c3d22a06a0835f21b6b305ef53712",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "4f42320c5595fec8a7ec9b4cc9ff2897ef5ed2b28062fa823931d909ef048f84",
    "sha256" + debug_suffix: "a8b6477c252cc867ced56baba40aa5aba18909f04af26160914639c036511967",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "7c48e8a76907551c0db9e1e1c3f33252d108ab3e173706f446a583822e080b66",
    "sha256" + debug_suffix: "858fb7b2cb58199a5af66e60fe995fba67e53358816ed99714579de59aa224de",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "ad148b84d7b41cfbebae8b31e9a3d37fdee891e8221d030b380cf85663c0be64",
    "sha256" + debug_suffix: "2e6c3c579b454ec85bc76c07a1df72570f076d916deaf3191b0eaa64d2fafb2a",
  ],
  "kernels_optimized": [
    "sha256": "273691a8c2487a246d84ea3906b7fe589e19d87403a55929e6a129bd0db52634",
    "sha256" + debug_suffix: "0654d71113508395a0d11ce369a46f730ef2dbd1e34becb3342e14593b146a56",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "f77f6948851282ce173e91e2c818618e67e64c3fb373ece8158c319254b5c8af",
    "sha256" + debug_suffix: "3ab6d1277f0fda3bc95da1116b3b81f0472373725110ff5dfc17290ba26dafce",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "db07954bde8d5cd06ed1f6377e634031b8aea5c3cb4fa074769021bfb4b1dbbd",
    "sha256" + debug_suffix: "207f20c18fff9c7c97e44ba6aa963c0886433e0e5ac91f68d4252dbafcfe4764",
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
