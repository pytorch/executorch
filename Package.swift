// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260812"
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
    "sha256": "146bd6bc28616ea97530ba3d08994717e85b894d6399541d2ff16faedf83865c",
    "sha256" + debug_suffix: "a7aed32e2c6ea7042cb03825549d0acfedbad85664ad91142d3722685173bc74",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "45f4e0ccbfbce6c189b9c37b07255817282fed6bec44af328cccce449900b1ca",
    "sha256" + debug_suffix: "60d58f06bdec6bcd2da5d28c0e5ab059b64ca6fa071006f907f9cfcd79a1d3fe",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6445a098defc24006c24889091703b9c55db4581522e1654819f88c38ffcfc5c",
    "sha256" + debug_suffix: "b327bc11114aa937fd6d145499de1e4445add39fbebf94dfa340e05a59ae82b2",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "c0b28be3994a4ba2bbe6c98083cf87a2ab0af0ba8e1a34182d88cddc0cb66625",
    "sha256" + debug_suffix: "aaea3a9c0dee3b0bf82d41a6f38420a2452e4a107e004f37fb9fa8acb0602180",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "487853977af0f5820def73ab5b36bcccb36f88f9dae60ac28ce7f02f156a25ab",
    "sha256" + debug_suffix: "ffa324290e111f69ebbcf8902c08b7de8fd33961c1fdc575643c5e4ae3d5927d",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "aabbb8329c94d4675f8f7a58773130ef76b56ba470a49ef7a0373427fa06cc2b",
    "sha256" + debug_suffix: "4300015b81d0383eb9eff6fecf45894afd8e8d106f2b1316a53f8db9455df172",
  ],
  "kernels_optimized": [
    "sha256": "74875a7136726eca11a4c09c28befbd6f6ac0fca07957f575a35eeba92732707",
    "sha256" + debug_suffix: "d6844cfc999decc847d490192d433153822f1e6d6a5288104b3ad2c0bf2108fd",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "db3fd24a8002ef0212325c85361bf56bbb75f55d36afc756bf16f97cc5258e3b",
    "sha256" + debug_suffix: "c3bac0f54315cc7fda65bac10a2db5b67fdeb4e771bf0c194db4c8a8f9a02e8d",
  ],
  "kernels_torchao": [
    "sha256": "e21ed69a0fc63dda568f40e720d83e765a3e858c427b1a69a695ea68d5304040",
    "sha256" + debug_suffix: "2ef39f56d5a714c2a7db231845f9f3c3ff856d3bf55a1fb0b738f94dfeaec5f5",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "966d2bd03a492010b5654c7a04862c9cef1ab79aea5fee69bb97281532de0b90",
    "sha256" + debug_suffix: "f4e8344bcc8e02e242b1a5c26f1c2955b42a3b5c4736ff7d987bf7944034e130",
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
