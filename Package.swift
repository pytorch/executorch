// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250802"
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
    "sha256": "af24d491d605ae41d8558e5f7663b132c841ddb45ee6e01012c6139777e2e9cf",
    "sha256" + debug_suffix: "8104e340f552db5dc4442a5140fb6d27608e6a65978d00b8d37cad3e760e6493",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "2297b622be18fe058cca0af0bf021dc2456b4c863023ff698f2b741aa61e48da",
    "sha256" + debug_suffix: "f1a4593e8f7668e96fbfed5aea53072ba019cbca2117f0261185c48ef70727cb",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b4b403de62c1bf869d1bfca96957284d1c62d3a813af2dd957cda484e352e8a0",
    "sha256" + debug_suffix: "ae9c112c84686cefe164a08be255f01edf7048ccf42ae4f29d3a418a08e3a290",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "19879125f44ca4de3573307479c7db266b111cfaee65829d73351ef257fb0e1a",
    "sha256" + debug_suffix: "7f0a43d22065e69870f46c133fad33ae05932c24c39b55f96769cff8ec795c22",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "b797fb26a3e21d25777e134f09f60aee4badda19c817e21267c9318f8ca38da3",
    "sha256" + debug_suffix: "aa0fde7b150922a48334cfcb5cd7086db241c4cac536ad56fec0ea05a1599392",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "cf435f892a3056b21c7668b681b1a226450272dda5578d72c1bb49ecc6102efc",
    "sha256" + debug_suffix: "66340c02a14e18b455ee9e5ef9577a671c677d262a1fd41cca0e0216e619b853",
  ],
  "kernels_optimized": [
    "sha256": "7a74331edd67b28007e7af3c3b2a56bc2b3ea168a82f09defb143b3ef2ad11dd",
    "sha256" + debug_suffix: "b9207f9d66be0b0621eb3c546bce953e1f653ffeb2412eb38ab34816126d72d6",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "fff54e999bbeef7879974112a280839c161f5fcc5a31977efbb1d8f91251a2ea",
    "sha256" + debug_suffix: "0869432a1b82ad835f76d9bf7d3288c85c094938b4aee2520dd21901103194fd",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "8a30ac1c4839dd9bc924566fc13917f07f14a40e5007f6b6a409b4e58e64d90b",
    "sha256" + debug_suffix: "4f8ac47b31c5841701433e9aec693b2279751251382bc7073fce5e47512f3d12",
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
