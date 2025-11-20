// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251120"
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
    "sha256": "e7acef586b155be143130a56efad0b5047be4163b30cb259a0b8f31cf0194117",
    "sha256" + debug_suffix: "ff3d7fd68086c8b1bb63df9fe93e7887150816cae42c29af2aadd824e6e28d06",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "44c24616296f9890b689566afa647c592b6263a45f4bebf154aeb8570a1f726b",
    "sha256" + debug_suffix: "387d86d6da287a4598cade8fdf3d6d38e80ffc19bf321c2feac812afddd2e01d",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8f1a53bc6953f36535d90f78c565183f6ff3a7bbcf854828090d83b175bc9b39",
    "sha256" + debug_suffix: "e4171b656623dcc60da76947c1d2dc600a1411cfcf86c7ba7159d82c44d66a5c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e110296c713237f32836b2a4fe9b16a0cbbe2c08fe15e96d5455bf3c474ba5ad",
    "sha256" + debug_suffix: "03aaec59f1d3bbb8f59f73bc32a4880fc2538a617c944b933a883215ac1a1996",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "f415e9a307e80ceac3c6d76b2ca184af30003515d5b4a1a978a2484579ece8c8",
    "sha256" + debug_suffix: "886cbef783432938c6f1406fa0901aa49c15272aeb7b9df13c0ccd2157abdf26",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "913e6880d20abfc4f50e4d3deeeb1777e6f5eba604a6fffc3ce9501b499e3030",
    "sha256" + debug_suffix: "2d6132d4e3e8e2a198223101e6f7897d656d6554a8a9a1460da185329afdfbb7",
  ],
  "kernels_optimized": [
    "sha256": "5368b02c85c378609d44b664f0f789eb4eb41c6454a963609a65f8794ebd40c7",
    "sha256" + debug_suffix: "295ac62894c146811a2bc94a7f08ad4b3ff1bc8f49233e6a5374c9e27890a73f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9ec7cdc04c01e9bbdd5ca8a17bad1350093c4e6bf9cf4f3b5180f8ce3799e720",
    "sha256" + debug_suffix: "347a4b6db05168da6e79ad76d8e927bef92ea593df52c5986fde706c7c9332c6",
  ],
  "kernels_torchao": [
    "sha256": "daad6b17f686cb2b8542bd3fec866b14a76d035546da71c29c84f87f0712d293",
    "sha256" + debug_suffix: "b7169d92688ff7aa69d9b937160e2a38d8a726095dca717f3af8d5da3a575385",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "813b2a03cac890b3c4fa270e8d6de5700afb352830a95d3e6d7479079b548bd3",
    "sha256" + debug_suffix: "59e59754e6acf94e1ce21c4631a97a65b26e96a9fc3c24fd715a5667a15d8788",
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
