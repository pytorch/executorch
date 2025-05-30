// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250530"
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
    "sha256": "e4d10e0c3ee080b5b24d70a0911149e7ff39a7b50c51839d4cc4e4b0ccb44409",
    "sha256" + debug_suffix: "b1d1efe636d04117e27ea9c7ae9a4e8e23442667f78919874408088b34239d9b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "af1b7d70ec5e0fc49ec834ad2169b5030db7968e3adb15f5684017c355954ae3",
    "sha256" + debug_suffix: "e1a74ca3e97e8dc6ec38249eb4765edd42c4de5d4f69317607336e9abdf283d1",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ffd409b6fda76c46be84a7ec2a8503cd9ed3ad105245f0776423b1a619997670",
    "sha256" + debug_suffix: "e6ef93ee5096e6905605f2805fbac30951f3d83e7dd6f1564e3aecfff844f924",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "b9b2bb718f061081b185063ff915be02de46085024362d0849cd7bd54da79d7b",
    "sha256" + debug_suffix: "f537478278985646139caca322f6eebb902efa1a0b40ecb74563a02909796dbf",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "a346f51c0fcb3620b7d2d44fe5d60cb168fc95e69a2ec3dfe4db50ba57bd7a21",
    "sha256" + debug_suffix: "58ecc024566acaa070da3b26b257024658f52a7ef4dff76faa095a2064b94424",
  ],
  "kernels_optimized": [
    "sha256": "78194d34de9ede4a480e29535289086a6011298499b564923957cde1a7e7110b",
    "sha256" + debug_suffix: "69e738f1b359c94eeea590b31239053f9f04080da80c40f5ee29ea0366c2dea4",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "3a41b986b3b0f1a7be726bb5c7bbd12d1c4306cb340105fee4619cabdce37b05",
    "sha256" + debug_suffix: "7495b1303e3488079ff1c8d71b13fd150db1a81b1167fc99aff8830d3f565571",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "7e47d0c73d72a4d6ced2c0b96bead3652c543c4cc3a34da3ee50387933c7fc81",
    "sha256" + debug_suffix: "9d13550dfc265aa71df4b02393ee4332cde595d8293f77e5657347968234265f",
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
