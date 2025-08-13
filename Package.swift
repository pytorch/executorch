// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250813"
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
    "sha256": "80a8273f2da9d49d6d0413e3c940360a8bec5f9bf42cc688317e72c3a29cbc65",
    "sha256" + debug_suffix: "4a3cacc6ecdeb1034987ee0e21ca83d87564f32be10bbeb178ea61ef0acb1809",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "e7fb25269db17f0fe71d31109d7791860f9572c219866a6d9f3314bb6770cd5b",
    "sha256" + debug_suffix: "a6389d3ced775fed1e6a7f36d4834e529c6bbc3ea0e90eeb51811480006c2f7d",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6ae2abed0dc5e1cf07b67024d189baa3d5e4d422ec9a48071c5964052ba9c7ff",
    "sha256" + debug_suffix: "2169f45661b209f59f12b18d3b15cffdfa973151746feeba211255870ba622cd",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "a83fe98045b532bd794e9d93ea1721d78846ae58a356903d0556ae990be51ddd",
    "sha256" + debug_suffix: "330b000bce8656c79cf0f90b35172fc1f39f0d17a9b448407c45b86f02653e3b",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "45bbc9306f28cd29f0a1c12c76ae38023ad957f7b97606887dc459d8a63bd932",
    "sha256" + debug_suffix: "b45dcc891fceb087bbf8ff2100f8e2e329e391ba7ee7877598912df409514f12",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "535d8bfe84dbf9bfd81e22eb970ed8a749bf8085d3a9989a9b02d2a20b9523ff",
    "sha256" + debug_suffix: "62449a1080e2feed4b706c24976c6534263697f719549d974209e98d127d501c",
  ],
  "kernels_optimized": [
    "sha256": "dd0b9992184791ef88a0dca0ef9d0466b4a31da4687d30ef2f70d026a792640e",
    "sha256" + debug_suffix: "adf3553caacda8de9d05223f0814fcec155deb5fed50dc00a62b5f5f49e91642",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "bc320c714f8c35a416fd1e427f2f5a251e83a33f13281f3be6d810114937b8f9",
    "sha256" + debug_suffix: "afafbbdf814fea4af2596b6b72e7b92a4ba2b400dedd4544e979d4998b04a32f",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "46921c821bb5d45a6487cb89b1aa3a60647b30cd7089490198ca98e1e9008b02",
    "sha256" + debug_suffix: "ad3025161c9973d35d0a59cdcdfb94053486e22f1fdd46159b6f05e223d87df6",
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
