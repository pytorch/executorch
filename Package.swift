// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260129"
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
    "sha256": "86d4d73f80877cc656945dbb72c1979fd4c5666aac94bc6dac049d176ba5edf4",
    "sha256" + debug_suffix: "1625d822fbdb769af5294b69aff59fc580cb107b1630ba65d52e0c81ba24d17f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a13e66558afff97fb22358c32795ff93eb5d6631d09deff87eeb0bb4ff7a2e9f",
    "sha256" + debug_suffix: "a6e5aad9db64ec18c56df823ee12facf936a4bc77deb75fd007ea70b57e4927f",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "4a6ec1ea2321e252e42677f33d48b32e9ed70f4a21e961d7fbc0a60345c28c93",
    "sha256" + debug_suffix: "5de07d7941e86e8c4f92e9774696fb8c21b73124862c39a47151dcf1e9ef1bf8",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "cdef9a62d121b2250587b947d13304bda0eb03d5d859c69a820601067fd12021",
    "sha256" + debug_suffix: "96e1df08af4ea11dd6df5675b788914143b723d3c97fff5d1ad299a47f1df9c4",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "1041c3e226f6e0726b271e8449548664ebe4d62c80f69f671a349ebcc85274ac",
    "sha256" + debug_suffix: "2489579776850cc1be87540bd8fcc3a124d86b91d93967f625e70c56ecbe91d9",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "08e076ed85a2bfb0607a55ce2ab5ec1b85fe8b60ed20a19b6a9c4d9afd3a4332",
    "sha256" + debug_suffix: "698e141701bd0322b1bcdd2e2485e7b31afa123089698f4de9cd02ec47a63dd0",
  ],
  "kernels_optimized": [
    "sha256": "a818d379d65f8bfd32a8c7406801f98afd837c039b1f253f94a81bd2b14022d8",
    "sha256" + debug_suffix: "da7edcc0653eab43d3064c51289cf5aba7ac00655ac5fc49f7e9326edca0aae8",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "f05f10cbb0d4db893d78811f4d8be00f676116b06cd067c6a0d24bf64e31791a",
    "sha256" + debug_suffix: "8073d52ad235b80532a8bca0771d970929b825797e01f5d5740a632276222e3c",
  ],
  "kernels_torchao": [
    "sha256": "7fc6931596918cb71d64f4d641ebfa6cd752704337c70771ed6527b013d1da92",
    "sha256" + debug_suffix: "e7a70ba1d818da37ca1d3f1d58ff55bdf4821ec669c456a0b6806d30589c7d80",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "f2445b1c47e23160848fa34433e0c11f256903596e7d52c537f253b034a23627",
    "sha256" + debug_suffix: "40c8fdf513d3f32346c2026e1618c3adb500ce7a759b3ec00994e25d32c94f0d",
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
