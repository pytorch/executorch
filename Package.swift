// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260217"
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
    "sha256": "0f44785878fa37351acfc392565f5e3a89e8e3cb0c925057ab4e837f5cc75c3e",
    "sha256" + debug_suffix: "6192840fcbe43db800df02985e5d0b114ff0a97a8a1799a0093906219356df3f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a8dde2c6c0b9e3cf6579cb26c17b1462e6e6419b63fdf265b8c2c01124e2d873",
    "sha256" + debug_suffix: "cb285f3200f1ef20c83dc873e57efe9ddf7cdef93a1a6b0724165f6d89371eab",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "805cd150f77e5b752375e17cc9acfb94af4a29997ed70e818c63cd098ddd235a",
    "sha256" + debug_suffix: "1332ffdcce35a3464051a7d7f50e8cf8ea0407a2c2a2a7d406992f0250fee018",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d9a422690c6a2fcc7b55683c6a398efad6ad0aa7e4b5342132377ccfc054beee",
    "sha256" + debug_suffix: "0de13b50b39721895432c511a6b416b68ca6e7ed23b02f3fa2a23d0ba9e53c4a",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "a8f1fbdcccbf626142a7adc8abc203ae92c9e1bb139d7dcd93f3591a8519a728",
    "sha256" + debug_suffix: "aac5b15c43371812786b22fd51338d21264649e0f032acd0063331ce133d08c0",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "91a95e8d58de7a6160829d2153bdb7aa07fe675443a9e84b64abb5ced8319a48",
    "sha256" + debug_suffix: "ddfb328365c953b3b1b057c41ac2ed375837b7d1006146bda9e7504bb21a6b42",
  ],
  "kernels_optimized": [
    "sha256": "fb856bde20777c7e8a5c8f5cf9a292e208cfbc9beef1f3ae2f4745de5bcfb449",
    "sha256" + debug_suffix: "879d12615af320c7fbc8edf1e95e06e5eb5a25237e267d50a71965ad81bf1f12",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "cde096129f7aecc608e3bfd4f07ff3f626a6d7d729d055de6b27b818acef2bc2",
    "sha256" + debug_suffix: "a9631f76ad06a682706370e6cd7f6c8ba237e642fbece4819c83abfa05a81e5e",
  ],
  "kernels_torchao": [
    "sha256": "5f939d82d2b400a730fcf3e74de2706d5bb2455ea0945b003497ee49b141ec98",
    "sha256" + debug_suffix: "9e1c423f44b0ff55e83362a0bd07184e0a4bd8631d29ed62b0cef5a5fbf433f1",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "f78835f5d07af63bd4b6e21ee5e12e21af9fe5f83cdc8986a19fea9c01968d3c",
    "sha256" + debug_suffix: "02d9d1aaf5efcdec2d0de0b69b7877f723bc19470de259d011e5dd30c65d1243",
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
