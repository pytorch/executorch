// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260821"
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
    "sha256": "8e3a8c698414af3ef3d690279f87452ebb043c0ef8b8f746d1992d1e50d9664c",
    "sha256" + debug_suffix: "173c4668a786c889cf3a64dafd35d40a2b09f630d1fcfff4db1521f8596b2fac",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "4807bf663bdcc033cf24d3477f73d152d60f490b18bbee7e9da306874476a467",
    "sha256" + debug_suffix: "7b7af4e0bbf8420239e96922ae70a73b1903f3411511df669dc22b3771dbdb65",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6c5a533381293e61f7326530c5a2ff998d237ad97cfe4325c4418a6755b98fbf",
    "sha256" + debug_suffix: "1f3d4ff3bd3fd913b55d09a2d67cdb27d5fd08de9e0b2fdcab57559327d9a43c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "41b7d9eab75a996b3e65d9b8879c5d7d5b0fe746df51a4a4093b68bc50427a94",
    "sha256" + debug_suffix: "dcabdd9d51859a4a4f67a60941930b2bf6e8cc6075d5dcea9dce84ceeb82440b",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "9a8ccfd23b8eefcfd8e5be7c5c5b5c404100337bbb50f1f701118546a1b77c9a",
    "sha256" + debug_suffix: "d3385cc019b928cd03e99a5a4635d59f559745ef2bd9b1146a8c2ef22b7d14d6",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "43b7b2767e84ef6c7bfabeb25f61cfb98358c7c9a6f1bb24528a129edc97c52a",
    "sha256" + debug_suffix: "335570120dd15693c8fd6aed09486319ec0d6be97024d459f675ae68345a5295",
  ],
  "kernels_optimized": [
    "sha256": "f33e0ab0ec06afe4456f3cac1d2190525b1c73f4d5b46d29e01756e85be839ab",
    "sha256" + debug_suffix: "31c6f561042d4100ff7c80b83f1eb858625a263a59a9eb814f0f948196afe063",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "1529b1980ed313874e93404107aedf7bdeca0b9536f196333a031284735ab258",
    "sha256" + debug_suffix: "be85d2b92f7ad34e8c7a8522eb9a1bfa7253ef8a80543e95dc05e62f00ce6cdf",
  ],
  "kernels_torchao": [
    "sha256": "cfd01689e60eed9a4e9e9982c2889ecb1818c2616bd6fe2d56c5b8b86a19e420",
    "sha256" + debug_suffix: "525062b57e329cf3e3da64f6b3fa3f4622a8c2d112eaace32b68b36b1b9648f9",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "0f18ec23654ec90f93e79e64802b4bc8ae601a051ff00146785215abd296bf79",
    "sha256" + debug_suffix: "d984dd5c4cd011e2f8f8986dc7af84f2c004c4612e9f703ce10a4b4a21dd696e",
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
