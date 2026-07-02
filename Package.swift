// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260702"
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
    "sha256": "6ad1a90686e8bdc3c2ec64e5a2dfb4a6119fc86274df37e2a83fc48c65506c44",
    "sha256" + debug_suffix: "7552bbb6ee44cb365324ed8a65d64cb10e72d489536aa2bedf44d9ae6c7610a5",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "ece99de210ff5225e97b488433d8a6174558fca413e14eb42c2a2111660b857e",
    "sha256" + debug_suffix: "6167649c4b6eb64b5d5dc852924428a4df6d41a93329d3ffc5fa27f06f3a5807",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "dad02d00002f090bc185ff12f09b7ac9a49d211ceea3e560c469676887e32534",
    "sha256" + debug_suffix: "652e109ea621609c9182e0a45e783df6ceac2a3d48f97129196a9f947c49e8cf",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "c3dd5d185216bdb9a4716c76dcdd7633d1b7614bf7460c3d69af881ed02c8513",
    "sha256" + debug_suffix: "2a4cb92f4e71b4e050bfd52d19f2fe8a99c4285aa3fc84f17bc3dc019fd82068",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "061b061d7c7ecce9e05664ac086353be824302f361707324969564f07521566a",
    "sha256" + debug_suffix: "91dab2b2fe3de432ee3e6df9604a9f5dcbc6831441eee73c16355dd850117fee",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "a1ec38b6eb9cd4e6be9b7fb064808fd3c153200a7595d5bac4a052ed184f95b6",
    "sha256" + debug_suffix: "aed83743227333bc97fc3f54a7336b5c008bcf684ee33af602720e9c25c3d918",
  ],
  "kernels_optimized": [
    "sha256": "ac255a0208e6366119c9d56d6e8b17e7e59588b741304d5f3a547a4103382197",
    "sha256" + debug_suffix: "a3035beab010c6d47c30487d533bad9f1c8664428bd25f8b43c75483b29416bc",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a7068cd8ea92e5e1506642e50b8e636f247ae895c7bd89f3f133c459c84bafd2",
    "sha256" + debug_suffix: "65c01ead8cd9346409ab9726c5d1f1c4a0ba23350bc93506c2d5e10db3768fd1",
  ],
  "kernels_torchao": [
    "sha256": "06a2982b55fa62a0ea12c3ee98986af93043793f1f5167db684c30cf2070235f",
    "sha256" + debug_suffix: "9b0ac71f109f85bb4751f4189e87b38b7052051aff2be98cea210d7bf40af59c",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "8a06fbc4d6e53f7cb18afff02dffb7046825767089c0922f092f40b3a1e83615",
    "sha256" + debug_suffix: "9d17a250d04b50ac67f8efa94030458daa93da7508bdef8ce60a9a7dd0d3fb1f",
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
