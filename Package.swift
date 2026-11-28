// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261128"
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
    "sha256": "014b45325fb66d9a9faf754608d99a19f5ef96da2ffb9eaaa9de2dd06e5b2c8d",
    "sha256" + debug_suffix: "e803bc93e839fab8d2a2251b585ec704779b23428c85846478f3679a636d711b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "b585f8f9c71062213a4108f7522560cec22a69af2c480b4e52c68dc0159f1527",
    "sha256" + debug_suffix: "40232710ee3681a7b8387db7d5f97a911c32525756b069018e183951180c0a19",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d69b50bddab5ac61fcc037c27577ebdb41c7a904be5417a26a0bc924584e11c4",
    "sha256" + debug_suffix: "c2a28cafaf20c34b78455f628582f3ab96bf0ff283badf8bb20ae0132354fe9e",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "98a45973eabdd600d6d81320da3bfb99ecbd7868db4dc62d3eff22e574c8ae62",
    "sha256" + debug_suffix: "a0a340014f1b8613c1234b7ae249d0dfef4b7cc39422e75c0dacb73a994835a8",
    "frameworks": [
      "Accelerate",
      "CoreGraphics",
      "CoreImage",
      "CoreVideo",
      "Foundation",
    ],
    "libraries": [
      "c++",
    ],
  ],
  "executorch_dump": [
    "sha256": "e35749a8684aeb8aaa5d44a1e201ae0cd8aa4bde632edb8aec6727ca487198df",
    "sha256" + debug_suffix: "82577f92ae36b41fdd4d3a3eae35fe940ad61d5b6286b3e9058f5264d8710292",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "441e42cef65ad9fec1ce7c21aaf5103b9747f0a8b64c4870fdcde047b33f9b23",
    "sha256" + debug_suffix: "122cb8684fd4a37206b85eb9a3c8e5924040097d84041b4e8fbbf09c9c8ad6d9",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "8d114f04c3b58dedab2d9ff8eaf84e64582f9bfaedb533a751c7b7d17375e8ba",
    "sha256" + debug_suffix: "65267ff07c2cf1cb6104c45dd491c4667bdb1612fcb5257da8deb6bb61435ad5",
  ],
  "kernels_optimized": [
    "sha256": "ea31ed5f2622101fcc171db3c01754b9ebed5178d7eeb392a09ad382c4329e87",
    "sha256" + debug_suffix: "1639efa780f56d67af548e6011e6c179618298a09dd410fa46e8a6e9d4b07b3f",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "50094fc26bfeb6dab6d330cf222b9d7779d20208c19f89f64d6d9591082e2e87",
    "sha256" + debug_suffix: "bbd4bfd90135d79042ed99dbee52ebd15a0245fac2ec5fcd5ab0b575e365c306",
  ],
  "kernels_torchao": [
    "sha256": "0e006762685348bca91da54619cdfdd7f51500f8a6c2b493ceae9c5cf50ae56b",
    "sha256" + debug_suffix: "e0d446d4a0bf5277b26137ba2f87b00316efbca94bedc562913a95a46f48291b",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "afc13afecd7b7d6e7b6532b9f7b5550737fc46eb43aa121d0137c8b0c95d2f71",
    "sha256" + debug_suffix: "7edc65d1941373cfcded2e64e0f947612a60344dd76e7a99707aa922cc9b0f0c",
  ],
  "threadpool": [
    "sha256": "d4cffaa0154a87591fd9e358ef32b6b777e7770f83b70865485075ee44a4cacd",
    "sha256" + debug_suffix: "2c47522c9c1ca8017e5eed450e8e23fe91a6838efbc568ab544bbd85338e4ac0",
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

// The MLX Metal kernel libraries, one per platform slice, shipped as a single
// resource bundle both MLX products share. Kept out of the generic loop above so
// there is one bundle (executorch_backend_mlx_resources.bundle) rather than a
// separate debug copy, and so the release and debug delegates resolve the same
// name. Each slice's MLX binary asks for its own mlx-<slice>.metallib.
//
// The release job commits all three files before publishing, so they are declared
// unconditionally. A missing one is only reported at package-resolution time and
// does not fail the build, so the release job has to assert they arrived.
let mlxMetallibSlices = ["mlx-ios", "mlx-ios-simulator", "mlx-macos"]
if products.keys.contains("backend_mlx") {
  packageTargets.append(.target(
    name: "backend_mlx_resources",
    path: ".Package.swift/backend_mlx_resources",
    resources: mlxMetallibSlices.map { .copy("\($0).metallib") }
  ))
  for suffix in ["", debug_suffix] {
    if let index = packageTargets.firstIndex(where: {
      $0.name == "backend_mlx\(suffix)\(dependencies_suffix)"
    }) {
      packageTargets[index].dependencies.append(.target(name: "backend_mlx_resources"))
    }
  }
}

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v14),
  ],
  products: packageProducts,
  targets: packageTargets
)
