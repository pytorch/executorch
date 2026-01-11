// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260111"
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
    "sha256": "b4ea5fa82b46be97dfb4420894256b05b5a33bcaa98b4100c359c38ad3d3b2de",
    "sha256" + debug_suffix: "c940e51f73d353c6caf0666a11bc2a14ad32dc3d8b74b3b4ccdca339afecdc65",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "6421b4b9d712ae96253a3f8b2ebea92fa9056e6f39732d0a43e36d4b6e76deea",
    "sha256" + debug_suffix: "2b35cb82f7428311f0766a52a5cf7c436a8b1be2237c981c8385aff4dcd67eec",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b003cd7fb81182512cd3444675c47d010da9e75a34df81aadfb8282887406f70",
    "sha256" + debug_suffix: "cd7ca0ee98bfc3058fb825e922d8c55ad6711adfd7b128dac7737d6a1be74f0a",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "b022f0d53d64bbce0f1abfae379b529e45359e5ea7ef33cedc9c44c1cd3bb559",
    "sha256" + debug_suffix: "ad2724a9a4dfdff83d7ca7f04f3bce5a47d4a4401b08f3d63d98dde20268ce00",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "82ffe70f48cdd96ec52e1de6bb42020df0884d0024c5d19630c18fd174aa60f6",
    "sha256" + debug_suffix: "b17c0a38339fada20eaeba6fc53564d169d32295f1030447fef822aae330673f",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c96a3679d9a6713c0f5c9f7f9447110219d4b1c6cb3111d44de73e2337aedf3b",
    "sha256" + debug_suffix: "e3c45751caf5dffddbceca71904f5245f893deb749cfc78e32d889bffe6b737c",
  ],
  "kernels_optimized": [
    "sha256": "133ef898324a1531b61b3726b0a4df6e23b04c6c599b8f43a64faacb21cebb58",
    "sha256" + debug_suffix: "e9f1ecf96e6c6503b295aecc96da32951ec14ca1b9dcf8a1a4c2c206553d032f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "139a3ebb964aba87f833f13f82d30a9a2bb4868cbae7a390b5ae442ad83f13e3",
    "sha256" + debug_suffix: "e1ea51f377c2aab8b6e1f07226c6df24ef3f943bf3edbcfe42d49076fff9a920",
  ],
  "kernels_torchao": [
    "sha256": "e7ce703ad8580b9fce7fd814a9706ef8f78084e14b25ad43b9437ab216f6824e",
    "sha256" + debug_suffix: "cb234781b6b1bcb70665eef4d6523fe3e05df8fd9fa0bdb182c72a4224bfbe09",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "166adde31d5a56f2f6764143aabc0abd40128360f5ed17f65d41dc7a91fec75e",
    "sha256" + debug_suffix: "a45b0acde154e10a430cfc0cd08397f4fa0d945725b4a33347312a36d7008d2d",
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
