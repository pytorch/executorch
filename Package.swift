// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260901"
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
    "sha256": "0083aac1f929e2abbc01d8a7fdd4b554b7ce824e1b367f08f54fe1953a3a5445",
    "sha256" + debug_suffix: "ad12ec9e94d2c79d3868ab0ae2806f2a32025bb21289e163f2896c15ec019037",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "fab7d0133389f9989da50c03276980c03b788618cb4e76ae98ad76a2fe4265b1",
    "sha256" + debug_suffix: "f7243d691cae07d834087e937611847720218596ed8ea7ede5d9bea0714c0b38",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "b8e04a8d82b127e1f352dd3decdf96864a2c0eb3dc6bbbd3816f2f926e6373a2",
    "sha256" + debug_suffix: "9f8fe0a1d3251e624969d1bbc54977958b0385958ff9711ccb56d4f8e91f9b39",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "264ef3c101a3689e4e36b6d750b0ae5c666ba16f14a0b4cb9f31257bcb0ac0dc",
    "sha256" + debug_suffix: "ce71e95f5b007a0376e3cb347f67ffe6d935a194ab10e7b50fab74ea8809ccb3",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_dump": [
    "sha256": "a45c0947e63b71d9896456fef9f3cea387af53a2d978db7dfa7d819b823e9efa",
    "sha256" + debug_suffix: "de4067a04c849d5d484d36d0d87da57ebab8c21344fd5f0a7d75506ab2c6960e",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "9499ca0a9f8bf49c12d926b736753476ea6c5ab03f7953f18a543ffc3bd090bb",
    "sha256" + debug_suffix: "01ff5ee05ae2b3d3bc6d19d5477c4890a1ae82b972c226ef5fad9f8c1bad2c27",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "267adfca2e0438cb8768d73d27154728bc7b1acc80c4b8278fa704b0d81d64bc",
    "sha256" + debug_suffix: "6984811bb7101e3cb194f909bb4b97ed3f77fbffa59038d84778d072a03d4fb6",
  ],
  "kernels_optimized": [
    "sha256": "d7d6a199abeafd3c4631df7921ef21f1cc93afe27268ebb44ddcdb9e31625c56",
    "sha256" + debug_suffix: "11f66161937df4bcec151a1551c1b735253ef12514c742c88093c160f2511bbe",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "4fcf11b1d12561ad0611441e99ba206cd261f4f60382748490c25ba90b4d6f7c",
    "sha256" + debug_suffix: "32d5d3072fec0e5534cb3f9d58c6ef91ff8d70edfdf7f30558fb6d4c900d68e5",
  ],
  "kernels_torchao": [
    "sha256": "2c40286da1f382d08b5e042c02681c6cf44d0789c8efe2fba0e8ec01fd455bc3",
    "sha256" + debug_suffix: "0f3809380ebf599669861c3152c9f74809d03f26b48cb7ca539b9e53aab8b718",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "ce3d884940f772bc230318f4812a00956efc13874f7e94ee1a6bffce70905115",
    "sha256" + debug_suffix: "ac649fca39e922be4b93da5dd9e3a04bc72e2119b72bbae88d2a44986647395c",
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
