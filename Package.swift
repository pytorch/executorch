// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251102"
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
    "sha256": "a38cdd88603dfd4411fb0747be506c796dd47ebf7652cbdd4e8adaeff90f6c52",
    "sha256" + debug_suffix: "b0cfc92020b9fc7d8682d4dae7da50f8e8cb9b3e097e5f1d080ebe7403e76aad",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "20bc86587698c81bb1e143212466a41623006d1c6f175073b327d93ac600c6be",
    "sha256" + debug_suffix: "3c888e9280a8ed278b8719488e58f6f6a67ae555cd4a3cce0985a78fcaf0f939",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c810d76c017f8a85b3c2d4688ed3d3650116055e3b4c193107921fc0a3038702",
    "sha256" + debug_suffix: "a678a23f54c6b8395ae731bc7c9aa706d840f7e28f7611ba229ac33e20d119ed",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0d1af5d026e6108e68cd1d7b5c033f4234e66d88a1e3782172abbe1d64d1f4f2",
    "sha256" + debug_suffix: "6efe1b2e400c4ca38f3052a1efd749c2aa900c5fce09ba5ae3298cbd9f3acbda",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "a440b1b59758542bd7e701a8dc5a6b3b9bd8f4b89186de3746ade2b41e467646",
    "sha256" + debug_suffix: "e7692511bbee3e52fe3d4cb591433877f1739d986671c855afc136bcd787f46a",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "fb385540ad66fea34b8a40ea19d88a0509f6e0bd9a8466c4a77ee4d38997fdd2",
    "sha256" + debug_suffix: "97d695f1e20b25f175d4014e87504c0c2ecb361ff41befa5f2f9cf5398596b49",
  ],
  "kernels_optimized": [
    "sha256": "16f6e257822e03ac0016c6bf4b600764a3551fcc6c61f80f42d937b8cfa0882e",
    "sha256" + debug_suffix: "4c6f2d8a55e54c41fb0d562a941716da06f013372fc42a6866bc592c3233004f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "66f9a5e03a63b30729b48ce1ae9feaf6e7eeb67cf8b50eb605a1205a97b073a7",
    "sha256" + debug_suffix: "e1a7abdeb28cd7d80aea6c9538008e28d74fc29b39fb0d199364c1584345e026",
  ],
  "kernels_torchao": [
    "sha256": "cb8912c200d8006e9e692c2eee75904c7eb872af2bbc2fa084eab3222dbc3f34",
    "sha256" + debug_suffix: "98b558865e284dbf3a833234042e9f5634e3b3d542d73619719c89cfe7cf5368",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "48ed784807c2878f171725c73cf8ec61fe1fff2d06b422fd21bcf4c02ddc93ab",
    "sha256" + debug_suffix: "2b29777bc0861b6f9a86421f63d4755cbfa72d20c4553d24c8818f4e2afdbda7",
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
