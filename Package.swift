// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251222"
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
    "sha256": "71a32ea6f7d1240c7c58aadb0fbe4e6a5bdc8e618f6afa8312936198fef0d186",
    "sha256" + debug_suffix: "1055d1909b1461be8a020ca46ad559cbb14bf58647c7c2d723a738118d043d58",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "0e2629ecdbd75295844314dfcfd5778bff1564ff9ea420f9b5697429db928969",
    "sha256" + debug_suffix: "f05fa6fed8f420c80206a1f832d8432f46d58812d50a877d79f03ae7438d487e",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c606959e019f0c64f488830c61e4de8999c04308b7db0d2ea97f233fcb03fbef",
    "sha256" + debug_suffix: "e6ca09ba96462d5e07c655f401bd9183be245b2e492c4ac7291f21a54e2bd2a8",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e1f48d6dbed205e4560b30360b4b1ce2b1756a88a29e87463cc2ada56007ccc7",
    "sha256" + debug_suffix: "77e16aa7c3869393429e080ff8fedfa7beb518f5db90b32b51d340cf9047ebee",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e461a70d23099f9bdcef4416a6d8d4ffff8d616f0b5cbed68cdb051eeb9488b2",
    "sha256" + debug_suffix: "2fc70d7692d0fde4e1a02d8063daa6851e191b00335e7261fde7fcbb3fdab21b",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "1669c8ff7ab8797474efb0108985d3221ceb92ab511e09a956021f1e4099e066",
    "sha256" + debug_suffix: "3d1529990b70205391f692b2721669e0693c3d0aec3a0392f4a1076fac9a2d27",
  ],
  "kernels_optimized": [
    "sha256": "550f1e5422d1c6570134f027e12913f273eb22edc758a336be193bc1907d4d3e",
    "sha256" + debug_suffix: "a2d9291eed61fd56642ab89ef2450d9cc72d9180d80dc509a3113355e619309b",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "25ea044513b09e5cd45d262557b484f2d827df2b4208a67100eb724070e087bd",
    "sha256" + debug_suffix: "d11d3b51351d79aa8bb30f4699dd277120ac7d698fd6ef47b7e2446e861e19ee",
  ],
  "kernels_torchao": [
    "sha256": "357127999d36eac4cfd08a9231e0f488c287c05f3492659130dd092fa96c866b",
    "sha256" + debug_suffix: "e05dee90c395c4178b937287b63108d9c99dd0d1f0faa75432de5f2562868811",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "5b333ed184358a8ce8f78c80a136e7eb8e23c2b4b01afcd6a59dd8e3b082b606",
    "sha256" + debug_suffix: "6acd493a854943dcb91e3b066b9d0d448399be37054d79d88cb156158f74e48c",
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
