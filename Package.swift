// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260818"
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
    "sha256": "1390f9b3bb681fcf9ee1f799935687f6ed48487e9970ed385fa47683f635f3e6",
    "sha256" + debug_suffix: "6d651c4ab2c72f37359825e6e6690da2fa2a5479f931fc3a4264eac161bc44e0",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "6274880d7e4bfc3c1f6017742583aaa8ef95e0518038238bf553193282cc36e1",
    "sha256" + debug_suffix: "3e768110c022e0c781cbbb332e1d77d5ca537299dad4c91ebc155d21b8fb57ad",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "603d76714a10513bdfa2d267a656d0dec7720e22f6dab79fd67fb7d5380bd8c5",
    "sha256" + debug_suffix: "79899b35f051c20af83f597db605911f8ca70c9903418819b949a7fd4c9cffe8",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "b05a6a0c619e9e1b77cb7dbe58675bfb1d56576a1433e07ae37ee34af32c941c",
    "sha256" + debug_suffix: "63be754b09e6a435641042372f60b4fe1cfacfc2da6b4ab6db8837d965ef8e95",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e64be3edeca973992f3f8168e96a172baf96777a0ac405c57f496116c884c3f1",
    "sha256" + debug_suffix: "b82c22674ac7f191d3eeb89b4abf4c8f104fbba5b8ebed408c7b0618b56999ee",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "08b4a9c13c563ee9cf8eb6a76483bb1fe71770469982de95d9f82444cb2fc011",
    "sha256" + debug_suffix: "b9bb377ead0303e4b8975d44264bd2ac1a349f2f0861dbef154996af947866ad",
  ],
  "kernels_optimized": [
    "sha256": "687de01de54cc7e4febeee3fdc4ee2adb81366fd09b2b521e6652d8be4d45a8c",
    "sha256" + debug_suffix: "d28e106d5e826160a849992a19596ae33c92fc3b5ca1658f9c3bc7f4fbec0613",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "df6eb0c51ac1901768989f5816d3cea4fa10f3d3c58a7d2e02e59e1d1234f4a8",
    "sha256" + debug_suffix: "0be4f2bed092c8a389d8a16c4bf6cc7b17d302c3c9b1e4dcd3c08b8e56de8183",
  ],
  "kernels_torchao": [
    "sha256": "6d42a04e7f6d63c2c19bf91d70a49f0a9c30c917e277da833af8e4e903086f43",
    "sha256" + debug_suffix: "5d36550b63af4573776ef52b7a37ee97fdbaae8897142dac1c20bd13c3866248",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "4c452b1a1025643d17a2a5e987d2dcbd3ec542cf5d55afd44fccdc6c9c608ce2",
    "sha256" + debug_suffix: "370e892ead56f7aead1a72541f62161490ef04a39489acc45df696fee53f209d",
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
