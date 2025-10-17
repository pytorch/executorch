// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251017"
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
    "sha256": "6d222242a2012e53fc4fbaec5025d3dd6775e98a93fdcf068af295b75a9b1413",
    "sha256" + debug_suffix: "4db021e6cb4f0716d7f096a7bc01bb14a480a22af168b24dcd89ee3a30c4176b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "acc66d15a791016ea2ac11618ae69bd89d8040bbcba724525d34ab9de2b03201",
    "sha256" + debug_suffix: "0e22d5729498b16cde5149f145abb6bffb0b47a3f6b22a973159f44b160e7eca",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3dba6e562e6edd2ad2322f1dfccf9c966ba024a20199086d8c7742c29f0b9cb1",
    "sha256" + debug_suffix: "7f478b23827e36f0d3517a1da256f6c08348acefa1f3c71a016a64b90f6ade97",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "7f535b0d2bc6784cb807319e6e984eda19c28638b2095837444c8f3d812dcded",
    "sha256" + debug_suffix: "a015d7b0f1108ef5f4f0c3f20e572e27e7ef5aec0be6b24783f1b6d49e603dd1",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "113c53563b40ab3852b262486aa355e61153e2d329218075bc2b82a1e9fff493",
    "sha256" + debug_suffix: "af89c2d8b43bb07dc82e327f651c158464f8d9fdf4cb4d7087c47f41c7b11e66",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "0854c5c9b9c45941a20be5ac26d628b10ec874121933090f3bf4d4b80cfdb50e",
    "sha256" + debug_suffix: "94daa97dcb6eac074f95ebb02d70f9de252bfe4f972b175188da21da99d2accc",
  ],
  "kernels_optimized": [
    "sha256": "06e60809fa68a55ad21e645007dbbc9737825a845077427d2a80c914d6ad80fb",
    "sha256" + debug_suffix: "fc5bdeb4c4a7f91899c158a0f94653841e2e84dbc2bfb9fa99b84656391a20ff",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b21229530911bcc7f7d985fe8784c5d42d3c36ecc789cafc85018fc01381b018",
    "sha256" + debug_suffix: "1b6031a65d5ee5b2dcb5b547b5204cf30490dfd53ae668c589921079ac0aeeb5",
  ],
  "kernels_torchao": [
    "sha256": "fe801bc9a102d6113fe191d868183f29ac57b7aa30a4ff695124e925e4f9e667",
    "sha256" + debug_suffix: "1ff447a9cecd3186b2d99804b3e9a7d8da628dee9305c63fbb9b4181b37c8fec",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e43090f5a336041a765e92990fcf039b7bc429b3ba64f5cf5c30a8a5c8a2550b",
    "sha256" + debug_suffix: "c37ec7c326dfe15dcadba0e6bc6bed0d8c263c92e73f0643c4cbc61ed6298a2b",
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
