// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261024"
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
    "sha256": "9f6d37daec0703b2a50c0b894a7fad3b5eef0a1d5f7fdcd48c600806c76eed26",
    "sha256" + debug_suffix: "db087c27d4d622b6ee96a0c0168b5a4540d790467842e4ba7fd4874dccb19fb4",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "7071e6d8781727ea97a21abcbb92fa42f267537dac8c79797a3449de0e19fdd5",
    "sha256" + debug_suffix: "82f9584788f0865ee453e5ab5c6f87aaa0004388234924f937ef9b5fb6997532",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b9d462f6c8b858797de52d962339502f83b9dfc6050653d2409d553fcf491397",
    "sha256" + debug_suffix: "586a1bba93e870a9c91c912b37ee31b317da1d2bac932d5ed4eaa5f11de240bb",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "57a9d721abb46ec14685bfc1965736ccd6ef7c8c722de19d4bb331b95ed8cbe5",
    "sha256" + debug_suffix: "ee0071c764af7b5f4e2931a7f4704be8fbcac5c3d2b1de8fc65b88117cca9713",
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
    "sha256": "811a1d8b04b31a047d5ec4294262d3b1f1f6648cc0074e67739f222033993981",
    "sha256" + debug_suffix: "882a1e6a9ed1166f0a81a05b9fcad9f9ebed576ac79d5cadf681095c68bd592d",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "3c7d86396b5c25f32d3bb141bf75867f2de447ff938d622738f05690b9f91e35",
    "sha256" + debug_suffix: "d18e0322b3da6b370888ba8ae3caec58a5211c8774bceadafdd5c1fd4c0193e1",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "867072fd3469255aa82f283edf903e042bafe9c63ffaf2aa93ef9fb4d2d02339",
    "sha256" + debug_suffix: "b3b55bdedc3514f383776ad9cf9bf1728ebfcb563c4c7611545c5cd47104bbff",
  ],
  "kernels_optimized": [
    "sha256": "e7aa927d8aa9fc3e9ab67b71824a6ee6a7e3c49af46f3f605e2cf12ed128d56b",
    "sha256" + debug_suffix: "da29ee630a57f05813df370cfac81f7cf3ac673c5a0140c7f14e5f94d03ef825",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "47d70b5dbb82a61c3ef7f2696368c72d708e211437bf562cfdfb9898575ff95e",
    "sha256" + debug_suffix: "5c93c70813cbdcee934e01a36437d8cf9eda5c90f73e84fc295af22dbb14a414",
  ],
  "kernels_torchao": [
    "sha256": "fe3ef9b29dddd0848a7e85d16daa5392f67017601b1ffc6e12c13d84b25c3340",
    "sha256" + debug_suffix: "b09c0edef5fc19ad65bf0895f13bb4c00892552bd27139170b07c78ceefa5552",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "4d5a4086e95049c66f2db7cc93c70f14f407e2f576f4f06c4985f6984f517217",
    "sha256" + debug_suffix: "fde4e5099d0b33076c61061583732f9b0530770f4e21b7962046ed588e39d57e",
  ],
  "threadpool": [
    "sha256": "2d36354888393bfa9883ecab1aca3e8774fc2440b33eacc8935817dbe7efa7f2",
    "sha256" + debug_suffix: "67992f43188146009aaae54c0d004f947a5756fbf3af1e4bc11e3bd4de64dfe1",
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
