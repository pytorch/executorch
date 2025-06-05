// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250605"
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
    "sha256": "a5e0d58b014033b00c20debb14784eea0a4fc625dff77b0608d9c201862a794a",
    "sha256" + debug_suffix: "3f7e809fa83a8eafacba1a81a2d30e3c6dbad61c34ffa3d8b2b248e33ed304c5",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "dbc22212f2f9f97e26df3b5b8182ee04be8b0426b64bb24b90db8de63f532eb8",
    "sha256" + debug_suffix: "eaea740efc49fc5cd31523997651cad15fc9c53b3206efd391a8433384b3d417",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "539fa2cd3ac17435b44f5d802c406ee5a09ff183921cf18565f0965fad45d378",
    "sha256" + debug_suffix: "27b829d383293f90f0c18bbd74e9a027c7c1da9676a19ab24fd922b35fb9112c",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "959823ec40a26e1f6d3919d7e316292892c86d73635359e4e99b96f5e33318d0",
    "sha256" + debug_suffix: "2cc4707063eeb8b569362a4aee77bb989d3f58dd9fd05ddeb3e25b4263802b6d",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "91aee9462d32931fd02e96ed41031f3df6d3b450a308bbbd05f741af3fa2e0bb",
    "sha256" + debug_suffix: "c1b43fdfd727f9de9f19272ccf088b6edf8e94ee5eafaff8c8c540e77314bc92",
  ],
  "kernels_optimized": [
    "sha256": "d86668f7b1a6d5cd3e5bf20e9de5a02c5f66f0830ebdced4d7402e01f4e04527",
    "sha256" + debug_suffix: "bf4a592cf41053cde710abe1c49c51a730f16403bb7b20c1c197eeb9888f6a40",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b1f969b33830fbe4c1c6ac5f40698e40b414e3b4c1da246c4768db382e9682f7",
    "sha256" + debug_suffix: "87a45c8ca2fce1013cc24a101bd538c5c5b95322c8866f6d4ca1a787a23e1741",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "104be8f6a7688cefb1ab8042efb80f7e697815645df48cfdc8c55e4568fc64f8",
    "sha256" + debug_suffix: "a18261fb57e452f41f361453e92a4f33d00e47cbd5b7d1d28de340b481df2bf1",
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
