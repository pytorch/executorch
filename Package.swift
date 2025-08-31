// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250831"
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
    "sha256": "8ce43e2a64e2d9057181d27cd54dd93b413d7996143a3b79b40988fe2ebfadb7",
    "sha256" + debug_suffix: "1840561565f75d8e15aca9784bb44e9adf56bdbcb557b06a778bc12487deb10c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "acfbbd1eb1e3051450d52b1ef4f6f16f7f46819ef70fd702baceaff50af66fcc",
    "sha256" + debug_suffix: "6fb256c76d20c263a645bb04bdebe6968476295a9bd78746f671697cc002ecf3",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8001b372de40a79dd2ff50abbeaad9dc9551469049c4978cd3fd4f89653744db",
    "sha256" + debug_suffix: "efb812c4e184cdb3684aa93ad9c9aeb357b8acbd5c88e029080ee2f07f16f2f6",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "5988f171aee79cc39989da1b20d859df7ba449e2200bb8e6d9a6ffb5f8d927b8",
    "sha256" + debug_suffix: "721027cacd7a1bf5ceb27241e57770a771015f9efe002904ee4c67d8f2610453",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "50cdc54c03aa13b745a54625fc00b443d5e6442087abd5e28d04207f55ac4bdd",
    "sha256" + debug_suffix: "0993a3a58554dc9414b70dc0c04137a9d03740f70f3e156b19e8e33ff56d9eb9",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "5d33a97e03ad6104d25c4d82ccd64a4de3f2fbc217893e39a35ff27b4246d261",
    "sha256" + debug_suffix: "85a8a03e5ce3262943dd4c3669a36c67e7f67120a7ea2bab23e3417fb0621f1a",
  ],
  "kernels_optimized": [
    "sha256": "024080bd3861494427d60b1221a3b8f0ae03919ea38dcef8fa0e594fa5588a8d",
    "sha256" + debug_suffix: "69d279deeecc71493d3f262aae4b253b6116728d7165bb77ca56da24903314f4",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "699dd53a7d5f76517aaa094ed9f5905ef98aba3350bae49fda944e943559a180",
    "sha256" + debug_suffix: "b320dd83bfc77c90dbbd465df19fdb10ec7a0e5762e3b4c770caa130b9bb46d6",
  ],
  "kernels_torchao": [
    "sha256": "4089f4729a89cd3e3b810837568182523322b888d4d7128c6c38a2efb0f52e33",
    "sha256" + debug_suffix: "305e3fc28c1b9b575c5b3080b96ff6e90ddcd10ea5cdbd569a64405f433206be",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e9e0e4d6e692fa89377d9a7b299db2daa0ce0defc9acdc9fee164bca19c77879",
    "sha256" + debug_suffix: "ddb87a2ca9ef540da31e127ef3f9b7bfd122742095b29605dce3993bbdd090a8",
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
