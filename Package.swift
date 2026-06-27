// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260627"
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
    "sha256": "b31887b348af612dbd2eb4f3b2625f52b1198ce3d4b4d9895d118dd1ff49e08c",
    "sha256" + debug_suffix: "bc69d55a11953a954471b4b33b1a4ecd232eee83f3a8424e03b08e13b9f549b6",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "bbad81a0129c65c6097feaeb6df09fdc5dd23da3cc19e7b1167fcbbbb17d37a7",
    "sha256" + debug_suffix: "6916979065cddb21c249102d7ee2a28da481402901fde347b7d08b54ea56fc9f",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "97a3f28a7777a1218140fafb5fbb2c74b3151131a1e1ff9e22d0003a6057f504",
    "sha256" + debug_suffix: "16aa11603803f7455dc7085c054a18593fcff5ee35bd2327f00c42bdb7abf352",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "05ea12f9c97a5e9c5964672bb0495c133c1db3c84ab0407300a9216d133bc7ed",
    "sha256" + debug_suffix: "9709db0d3b5e3d7d9cd2b01caf1bcc8e17890742c84ab26a6af30edc93081644",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "95185b02e83c09dc0f536def207a9f8fe06063ccba620bce817cea8b6e1b45bd",
    "sha256" + debug_suffix: "0f80052cd0cfa7ccfe4b2ed42c2e72bc0073f0ee97421e10c93df2e45780e71a",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "ffc188aecaa8c49254e2526b709791927706bf7064e147116411572416944d72",
    "sha256" + debug_suffix: "4f983a37fa07689ec7fd580a4f41c14f629da69660cfa2a534274d6158049483",
  ],
  "kernels_optimized": [
    "sha256": "e0b8e44c8e2d7e5fb925dc4f764a79d19503e197a2245bfc6cc7453ad100396a",
    "sha256" + debug_suffix: "4af75421482d3412d5213843400cf180c75cc590ea363f2ba628412bfaaa0352",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "c247135aa8ba5e57b2eac6d364f292dde66ad988b25386fa98108e5ffbf46a20",
    "sha256" + debug_suffix: "0893085b01419a11f54796b9943e40d80c4424de4a4a9f99c93e60c2f1310efb",
  ],
  "kernels_torchao": [
    "sha256": "cce02778010c840a716be11b873006daabacbfb8e99da304cd5c5dee71da7fd3",
    "sha256" + debug_suffix: "ecfada79d3ec905f5a0c36ab48f6b818a913f6abda5d613dffa6f00df6bcec26",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "c0d5195a66be358ef4fec00f287f0ffdc3afcbbca420802d4ab0f2e4eaeddfcc",
    "sha256" + debug_suffix: "c51b4bb8af834da2d396d527b5a44fdcd6aca7786f9d376bd17244e71dda719c",
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
