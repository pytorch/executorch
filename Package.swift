// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260616"
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
    "sha256": "b00c5ebdf9985a80f1293528524b5b9727eb8ea0e2cce0489094603d8c382757",
    "sha256" + debug_suffix: "d5faac5be679f7515b0c3c92dffba8940e81e106c1c6f95f4248ff83ef1151a8",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "fdc99175a140ee6e87faa189db6695cb04baaf1e2d1133695966b839d6c1b12d",
    "sha256" + debug_suffix: "aead14a72fe403965fff46d2018634adfcd4a93a968a7207ceeb54c42730f777",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b63873cac3f0b33311cea04c0ce7510d3b99fd3536ac8d2479732d32dc97406f",
    "sha256" + debug_suffix: "6c6f73705f4b3f32f35969d7b9e5249c9574c664319626a92f5da9546fb65abf",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "c7be646f0f38cc205b977dc92654323e2094a3d02339eb4fe1710ed4d27d0a68",
    "sha256" + debug_suffix: "2313354735299a65360153039385a1ede031b3647f6064eecd4137f6ab4441ac",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "b2ddab358e2afd1e16c4bf96cd1ec3cec2de548aeda08c483f7a55f7a6a0a812",
    "sha256" + debug_suffix: "e6f770c7db0fefc38fe9e42dc00d19ed25498878841d2e86cedd3a8732e8be79",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "a64c989f4fdababd1f9c94d8a58298d4b383322e939d3a65d27d92cd6970ba0b",
    "sha256" + debug_suffix: "93acc0f9295f9c8c988a685fd75134ef65a59e33379a88d4fd97ed1d5f6a646d",
  ],
  "kernels_optimized": [
    "sha256": "108049a14760a1542e180dbc2309bf8d9dd7407029f1a9ee0eb795929779062f",
    "sha256" + debug_suffix: "2ed5b3950cdaa13c51b3891bd575f7be86ba53721f951605065ef8eed2bbabb0",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "2e6dcb50cccf184a7b5f2f974f62f729651a03efe784e70f41a624266cb27f1c",
    "sha256" + debug_suffix: "9550dc250d12f5735f39aaf79138884ac79c80d00d8552e0a297c154b6a0b10b",
  ],
  "kernels_torchao": [
    "sha256": "7d29cd7af63fcb5743ed851bb4366d19bd4c318f4f074b6b1c7b6333dc2283de",
    "sha256" + debug_suffix: "9f4c28879bd4ee020d05e9e28746a82a629a988f5f183de48e5ea20aa722cc56",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a467475addd46cb6e81dfacd16bca58e58fb5fe7ea6ec3b826262558089da19c",
    "sha256" + debug_suffix: "2261ce6d00889378f976ba90b4ab87e5e56513708c8c7b1b0cf9dcfade27617f",
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
