// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250615"
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
    "sha256": "52dbb56a128b95b63b9c9bb2a2bd5f1a21b5dc3b81102fcbe163dc9b89e87426",
    "sha256" + debug_suffix: "7f01cd4225f4bb852e48c1265a0e64ce09c51f25d7ee73127d5dc3b85d41c2d4",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "72d5048e9a39dec7a788a5bdcecb1df2fe2e607256f4666bc78f2ea547b6fe56",
    "sha256" + debug_suffix: "ba7c6188e8850711c66d19288062a8e983ab2723796af2d723ada956ad94cc65",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "667e726ffda1c27eab9bcb225d8cb9b625bcc1f1647fe304bbd8d7db8992ffaa",
    "sha256" + debug_suffix: "90ac49cd0f86128edcf197937fe47f7c3583444f1d3d0ff51730ee54e0b2c4b8",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "97e14148d19f29a0fd6dfe5608fdd3330c91689060ca5bb0a90ac61870e55146",
    "sha256" + debug_suffix: "9aa10dbb5703e9d068237005bf889ce63aa73f6d9bf95ce2daaeab6041cb8068",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "52391d966866c41cfa3b2420d6f41866b3659948e926f72d9c1b7f4984dd5854",
    "sha256" + debug_suffix: "78f41213bf5c304e31292cc14dabd5b3ac16792f7d080b206d70323ca1cd6374",
  ],
  "kernels_optimized": [
    "sha256": "3f61a8b6c6c09b23c357bf515b147312195c16eb4a7e55ae91044aad8f5f6e7a",
    "sha256" + debug_suffix: "c608af03a927de2315bbfc78723468a1510764010dca7b3b9ad2a7d4d7875998",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b059bcba0b1929f73599697881159663bea64f3e8eaa2a33ccfec8a636cfa911",
    "sha256" + debug_suffix: "acb9dd893f0ff8fcbce040d306bdb85191563d91fab6865def56e8d32e88a978",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "3ac6f0956e3e80596aad032a1983368c8ed2fe0b37a415dd501a63f134936ea2",
    "sha256" + debug_suffix: "646c383ad48eb36d5c8d88cf89db88f85580ba8cf66b5fdbc9644d2f6e400ef8",
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
