// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260514"
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
    "sha256": "c2111da12aa47c936177d0b1d436480c98e007f502503cd3fbfc89159fa26137",
    "sha256" + debug_suffix: "05d46fd8fdb7e5ade5bf871c92bd2812e8d8ba641e7cf04973a38597ae9b4cb4",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "0310d6ed1dbc1b70d518d8fa00ef5d3ad6649975656360c1e608f2fa74a0c960",
    "sha256" + debug_suffix: "d07030c903fe83edfe1ebd7d9ffee6011f367127a94a5152db5da94afc7cc274",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6343d17c612fb9dfe9b32048b953f90183cbff910b25c4cee0b252d8d51129a6",
    "sha256" + debug_suffix: "fccc7f4ec72ddb03a68784ee5929f3e02d3b06244cfc00eb86c4a38708bdf703",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "51e5e23a287f9cb7ac044052d274551832dad0319b565abb934c858fdeefb100",
    "sha256" + debug_suffix: "0c1e9b499eb76452f8179a3a44157dfcaffb2e282294ddeaa3589094805723df",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "98f8a847094011bd0b6151182477d28be35aa84ef0d673008a4c9a1b286ee702",
    "sha256" + debug_suffix: "a1ae7030b1192d55a8bed851357c7ce6094b269283e420784770ec872d36fa12",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "2d944bee171500c8228b75a79834d80ced468484a7be5ae3c33affe59a7459d1",
    "sha256" + debug_suffix: "398e8103884f4cfdaf0e60a2dadc51b49b8293b22541323eb9e743ab8a17c286",
  ],
  "kernels_optimized": [
    "sha256": "75e567be40df810a1383a3b52b486256076ed7921edf2a60f1f16eb99a4ae3ba",
    "sha256" + debug_suffix: "7da174f970d29efcafdddbfe594aeca701904b304d161067282b6779bb1266ee",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "696e26229126cbb11518f48a11e28db13cd997d2bfbe9970e176c5a53a79d424",
    "sha256" + debug_suffix: "8981f697979566a62acfe4e54bb8ef7dbefb2c26ff6f37d2381c51710bdfb142",
  ],
  "kernels_torchao": [
    "sha256": "89e3d3eeece9de99e6b9e147be44bc9ea93be3532038a7cb4c89526a429b06ff",
    "sha256" + debug_suffix: "a4c51b7ead9e1613b659026225b810ebb3ca88295fa8be39f51b42ce9609ff8b",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "b4e0da08973000c796939528859e3a8a3abd4ec677e21d385dfaecf1986f435b",
    "sha256" + debug_suffix: "4d4b90c1385246ba6bc41c5961a2681c90c7a5cba1e27bf10818504f9195ff29",
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
