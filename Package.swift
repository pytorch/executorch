// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261022"
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
    "sha256": "a97bca58874e1b086149823b9e46f6c32f134c54c46ef9381dd2acc585b9747f",
    "sha256" + debug_suffix: "92767ee89e342fd892e29538a0cf5742bab0883f5495dd9833d03cde69567b63",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "6e37d5f30080874aac5a44978b782836ea7bafd11130a28e20ac511897636143",
    "sha256" + debug_suffix: "6eaebcf13bbea65e6b2f20393eea9dd86f7adb8971b56f6c46d89a86f47bbb59",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "629c5562c77b291715328d0bbee19c02cb171cc94a291d65fd4827825fe37c89",
    "sha256" + debug_suffix: "f0ebca20a49ee5c90957c0eceaa79de07aa965c61af4d9cd8f64f44f7594aea7",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e3b2ba7ba4453e11116424726f351cb6a88b803091dd249f234d28bad27ca343",
    "sha256" + debug_suffix: "ab4271d6eae5b9409a80193c770ca87bbddf3076db3f1f3f6344f6dcd56cefe4",
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
    "sha256": "042e8706eaea097d01a980722081011d2ceb02739de267aef19210e0bd42265d",
    "sha256" + debug_suffix: "81c924982e5a92732a6d3ba57bf672e7c6f77aaf52796dd596d7ff97aeff2f0d",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "e615c55b8a423bc5c5d640e2c333c77fd8a8130b44e293ecb28e7cecfd975d34",
    "sha256" + debug_suffix: "76ddac8e7268f3e106059a6f8a6c7e7f738f9e3fd7164f5145ca2eca4993e9db",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "ce0070ed18cbe2927d79a1f601ab0ed9302e12496c969e3916763f4d60f99784",
    "sha256" + debug_suffix: "06bd0f67b6a7a9da973af417377338a143ccaf7acaada4898e21dff2c68d1967",
  ],
  "kernels_optimized": [
    "sha256": "4ed9098a7990d820baca525e62567fc65c2c9eed2bf9955dff0b8c4e12169176",
    "sha256" + debug_suffix: "c92b6b92ca733b7230d556cf0bea8b3843ffbf3709c26c9bf99c1d3c5136a794",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ee6f7938940a614347c8d4e9b2bfdfc8be8bc3600b7d3ec01c011790ab8a01b5",
    "sha256" + debug_suffix: "0de78b32996f98fdf5eed9d18183023e45a68bc129def71901ba58432bb1a257",
  ],
  "kernels_torchao": [
    "sha256": "34fb47759366a18c57c0157029a1efcb7a42b15ca1360160ae3e6bfbeb8c15cb",
    "sha256" + debug_suffix: "da0d076c56c15dc0fd4cc6e486989fe0b27b9b15178d3c8886b538d80ad50a6e",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "bdeeffb6a06e841945bb6e34a260e334f17ab7faaa1d5510cf9e3a214ffcc568",
    "sha256" + debug_suffix: "1548cb1dd998934a6d1447801129e4e1264baf9a7398bdc70270cddbdb3e058f",
  ],
  "threadpool": [
    "sha256": "43b561f671a41ae85f2acfe6ee8792ec62d4ac0df8d35f9de9aae69f88b153bf",
    "sha256" + debug_suffix: "637e760dc7506445de0bede8f39a5cf2f1da2317408ac02e7e5b05a3df585c59",
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
