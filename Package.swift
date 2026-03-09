// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260309"
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
    "sha256": "042582920f247fdcebd11356c610e098637b4c60a7dca71b0bf77bc00770ab7f",
    "sha256" + debug_suffix: "f7688787497cbd66d43dd05d5e5c2c9797c4ecfe5b35ff6b91365444724576d0",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2968ccde886d1453d74f446ba915156c0fc95fba65db7ffd7aaf58f44f291b62",
    "sha256" + debug_suffix: "d0c011232c96098e242f222ad97690892c322950af8f8f5695b29e56abce1ed1",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "dfbea8af7f1784d5fbcdc8fa4ff5f1ba7b10753a68e1ad7a5b0d2b839112699f",
    "sha256" + debug_suffix: "21c699f3b52127eaf1236975323f081226a8e30fc1f62950929c9250b825b9c3",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "b0c48774d8dec329e4a5628f84810813fadedda97f5dcb3ea1c3e6d3bee5087b",
    "sha256" + debug_suffix: "3b4f24ece9b8d016eb7b4872eb2312e41c2436ce626b47a17c2990a160d2ec10",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "0bd371abedf9759676c17cc37af590505e61a9a0758e3ceaea6a894e866dff26",
    "sha256" + debug_suffix: "aec2ef421a24cde3aee12db9a6074c4b34e9f64b74d46e91b1219936404ad946",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "cd322361aa9d23dfb38f1db9684c8038b3c1bfbe82deaa00d83a9d64d1a65405",
    "sha256" + debug_suffix: "32dd1831ec2274a5fc1efabdc6e5313ccec087990f6d548029b8f9e1856d2ec8",
  ],
  "kernels_optimized": [
    "sha256": "e2c55816f2ab4d49b3e83639d417ed61961f6870f7e693af14c353ca26b02f00",
    "sha256" + debug_suffix: "97a34c0fec559ae6b805b191f97fdd975f05ef7d8e49e2d90b7e33b56ea5b334",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "5b93236911b7882ca4508bd4164471d7ae5df242f4b45cdcb190099fc5a8d3fe",
    "sha256" + debug_suffix: "2b3c5c8d757974950431046a6cf988d6b104c7277490dbc65061f56a7521d678",
  ],
  "kernels_torchao": [
    "sha256": "d532772f5f53dd7a633324ec665332b172034ab76c161aaaff154a4be0eeb60c",
    "sha256" + debug_suffix: "f251ccc8aafaf63d46e9d7a03c29615ec1d0985715be4e7821ec3c2eda1a49f0",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "09c2defc1381647885722e50f16eadbff9efc9bb7fc2501b5aa40641cd0d16a0",
    "sha256" + debug_suffix: "df2899f5d9501c4f54a3b4f6a2e0cfe7d6fc71dcbd7e1126008dec80aa83e312",
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
