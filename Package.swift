// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250714"
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
    "sha256": "5cb8e0141599e90cdca4976ed64c1ca82c3e2066b95e00f58c14f0a28ae38786",
    "sha256" + debug_suffix: "75ebf86e07f980ebdc8a1a52e759cb150e578ab21be8d27e2c39a09ccf66390b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "f2a1eabc6220ca4e86feb3bac773c7248cf9ac265d7959bc2413f96c695150ad",
    "sha256" + debug_suffix: "a72f4c9d164548fa3f77dfaaaabd1ddb1b96b5d8772f4543e2e683012420c5e0",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8af60e397e91ed5490a0425c78ee96628cf5d18c413c096971ae0f0a22515f17",
    "sha256" + debug_suffix: "0998f5a26ee0f517d63f4fb9def95a645cc95d6559f778fcb6847b267043a4b3",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "53398690f7b5c070b1bc28c616f6087a850ff57e0e8a40e2bf048e35ec974ecb",
    "sha256" + debug_suffix: "01aeb222bb49884d30e1e8b97900aee18a51f73f457bf920ed26af24baae96d2",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_llm": [
    "sha256": "d520f6df572c1e81156ced4c1b9aef8ca5eb85cfbeedb8c9ffdc2a4c6dcb15b8",
    "sha256" + debug_suffix: "be1697678be43152486ee2b6cb007c3ed7904b799e78af371647b8067630a14f",
  ],
  "kernels_optimized": [
    "sha256": "790dd2afe37f3b96f769038c424bda72e6cd6aa4ae1a4d694652277cacbc7f75",
    "sha256" + debug_suffix: "b3e7d101d9216c094531a4298662cecb4d94737435bda7866c895fce1c18ff24",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "149b95cf9cf6ac8d49b9a35dc588ebb9e0d3a6cce3348c178af359d6b7a0f4c0",
    "sha256" + debug_suffix: "775b3f117453d33705f2ad2f805ce0be1f78dcecf30d723a553d73a0f579fee1",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "9270fca3b8257539f51a335e56857c03d233370bc4c2cc0d46068103c8ebe695",
    "sha256" + debug_suffix: "c382baa631f8f5f093da6be58c4afc37e7fd92dcb3579c81b6c908032f353794",
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
