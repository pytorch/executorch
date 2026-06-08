// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260608"
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
    "sha256": "7805ce67bb3ad4d86882b476bb1af5f7c2d1b840b24097629a1410a7c5403b36",
    "sha256" + debug_suffix: "0ef77706fde54fa19c118539796e03a47e8214509df438e9e6838ad6ef079fa7",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "447a11e84fba5ac9866ccfb60319d9cd2fa75e7e6cb161eefde1648c3478e076",
    "sha256" + debug_suffix: "b9f6d75ea8ec810a62014897ea47eb209e084ffa1a8157cee03e1ef514a7983a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "acb4b820c680937820e1252a1aec0895d8451a60744b72070aeefcbd3206a2a5",
    "sha256" + debug_suffix: "92741fba308e86924306343d497ac18f80c204abcf9740dddf2f70ab9e65a86c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d23f616f032aa71b5cb3eec7170165cb1d3a3dd4a996e342c02bce035ec70a15",
    "sha256" + debug_suffix: "30574a1924850c74d0315388572999856b7ec64b4c75d1c49dd578dc1ddbae10",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "0d73b14621604c2875fe725fa73d96bcf29ca332fbb72022b24401b9dc4b04c4",
    "sha256" + debug_suffix: "8523ebd15e027594f8bc39dde01f3f3e8234f0b3abf21f5364bf109273279dea",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "3ffab48b27b14f03a49aca344b6cfafb02be18ab25a159fe0e47c118ca6e5b32",
    "sha256" + debug_suffix: "5283f10a2f7f218dd71f098a8dc13e47d6d80adbfc457523f1619bedeecfd7e7",
  ],
  "kernels_optimized": [
    "sha256": "86f037269ba4fba4163386cbce493b4c1dd8a1a9a33225e6a52926a36f39f5e9",
    "sha256" + debug_suffix: "d34fbfe881d4baed99d80a387786d0249241c48e171549cd8edd2d3b7467af57",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "d973ad831efacafee8f22bcdca45a54c103df03a8726cdb2094e3ccb4fa9ee0e",
    "sha256" + debug_suffix: "764a535ce2ed1d2a76461b79c2db464f72b58d22a206e48f9b1440ad420c57ba",
  ],
  "kernels_torchao": [
    "sha256": "1947ef79d9196fbc592bea610ce96329748b486c907ef214b4427f82aed1b1c2",
    "sha256" + debug_suffix: "e9e60405313824fe2adbbeb454e56c20efb3f1b563ed71659d6ce5f2d2b6fccb",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "9a17b594d9575fa52c82b936a3786fa026f968772a01a69fc2d6409a17b87193",
    "sha256" + debug_suffix: "ecc4a7674ae597544988a0d8305ac3f400a485184f3fbc5ce7c42037406f637e",
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
