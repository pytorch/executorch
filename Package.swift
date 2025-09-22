// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.0.0"
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
    "sha256": "99bdbe5a27862df82bfac3e9fe36f50e28b20e36afb29dcae1225b19749c997d",
    "sha256" + debug_suffix: "2f25784a9396031bae5df22e801d05577c7ea235721d8e794321fb718c8b5863",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "b7ad407a24ea304921ffab3045ad76534f3bb34e84e2e2505a66a36c43c95932",
    "sha256" + debug_suffix: "96fc5cc9e41698849984dddc11d8e1d430fcb3d5ed9d3d43a0cdb28930d11d07",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a8af701ba6639d950684aaae509862d2877f5521483d46964d0eb3850644b616",
    "sha256" + debug_suffix: "1194d771a310fcdba90bdb17271055b5d2686be4bad262748a403961d36bb356",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "9027d633524f650c9b1bd3b09767f85a45f884741a0b76aafa329b3e0da50402",
    "sha256" + debug_suffix: "c9254af94fcda370ec07dd3bed8f13e7d5d19d06861883c3b1edba9bddfd05b8",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e83ebf011fac4c15b254a04b53c801930acabe705dd8f11f9a4fc38a1c086ca0",
    "sha256" + debug_suffix: "4e312aca1887126c8928c1253336f3915bc2b3b6fe275fbf23274ad4857bfdff",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "104c77bb6eb6249764e452e123e660b9f3213bf99b74ac3564aa19db2da4b54d",
    "sha256" + debug_suffix: "2b81853b6fb20676924985e67d6e1b2e54b9252fae549c4667231b022c18d67c",
  ],
  "kernels_optimized": [
    "sha256": "818a8bfc3455434ee51df4d76cf97e3cda7452ecb9d62fe8ce3fcbbc6c59243d",
    "sha256" + debug_suffix: "db14d8d21b6a4b4c3ace3619d7b9e575b46bfbfef313258f612a38d012d947bf",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9397058d29fb1b5db322fbbfd384af8c1ebdc9212b3ad7411e3e3c5da26bdb59",
    "sha256" + debug_suffix: "f63456ce40bfec72417440fa36854a151bac54facb6137b0cb9c2f64f6246f19",
  ],
  "kernels_torchao": [
    "sha256": "46555208bd8681c8b693232b64c5e23138710cd24e7902137253683ebd4a525b",
    "sha256" + debug_suffix: "ee6d602281702af0a2c975b35c57df2c01b419d5b42dc377438cdff95fcf45c5",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "eccfe361725359e7d789c0c5d1dd380f1eb8c34c1f0a5cd23142ba59bf7bd689",
    "sha256" + debug_suffix: "9a6208116efb3843dd243ea213775ecc4d5fee0f87e29cf20e5fa6822e66809e",
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
