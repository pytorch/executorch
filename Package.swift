// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260625"
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
    "sha256": "6de1339895c04906bb0815cae4db5d798e5bf7dc07575a8a63882f54e53e9810",
    "sha256" + debug_suffix: "f5729ed98be81ad0e63096b04b2ab8ca7021fdefe2155297e5db66d201d239fc",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "e392483e4e05924b59ca2840fbcb9b56ceceb52c4e593ceed088c5d130ad71de",
    "sha256" + debug_suffix: "d279951ad1725248f6473f6b68a36fbdece6631cdc25ab988295e0ac8c645955",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d6c9e7b04fd2d9f4c807a83c7507ac6449dc025c1b9bdc304f24e979a2590dcc",
    "sha256" + debug_suffix: "3f7632ebb14b73af38bbacb7797e2d9ed2379b176cb1c9bdce96865a2b122e88",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "6a5f378cd18b8f76d65a287b17a6d426f8ce8270326d28c0ad93ed0d6f48981a",
    "sha256" + debug_suffix: "893e80c33ac150d7145c2d9fc43c520d5a45b01aaa61e49505b99ea11c67d170",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "d6234f6c96ea22ecbbfeab93075c543dbed0afc33362aab43cbdb746943987bd",
    "sha256" + debug_suffix: "e9293f15573df2ecc403c3ad0dcccc0fa067f404672fd16cb1d2dc7c907590ee",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "9fa081918b2f0e9e44c88713e479771c48e7b1c1e55125199c5e6cc88e53438a",
    "sha256" + debug_suffix: "7569b7aa70ceaf14358d19269a705e9a67da53b9119169117961bf4044554ea0",
  ],
  "kernels_optimized": [
    "sha256": "f804a9ca39deaf5e2e49783332f9fc80a32457534c7016d88febdc7074720c3b",
    "sha256" + debug_suffix: "d4dce22108d5c5b544ce0adf1db6239696fa9a6754c2ee94aee50a14a3d41214",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a61e3e19979b4fa38f281951690fcd56735a74537faf9471a47497ddd8864092",
    "sha256" + debug_suffix: "5a3f5c027d21b5a2c32fbd3153e5f2c7cf9b800cca20f877b7da1e5eb0abd0f5",
  ],
  "kernels_torchao": [
    "sha256": "fa42d937954fd6efceb8992441a273fe6f2d8c3b3652b32baa4fc0ab86638620",
    "sha256" + debug_suffix: "f4542a89fe1c46de299fd31900a87b38ebd1a60959e99e5ba5c96b5d7e5dded9",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "99d11a9e7233255d957cc9d98bb1be4d46442fc3ff2374fdb8afd66714704420",
    "sha256" + debug_suffix: "a5d8f8458a4f965240589b9a9b65fe3d7841950d1b9f32addf502a1e5b374664",
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
