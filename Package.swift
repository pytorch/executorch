// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261103"
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
    "sha256": "9cf474a124e8d169e6cb9665dea4bbdae5069f7633cb16d753ae6010d3e980e1",
    "sha256" + debug_suffix: "84201d0a7432563d6f840f09d50acb9def77764adc6ad249a0fc9a3088d7ea1b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "b5d6d742ade9370de49e2019ca0a82724342ae58ea86724716c03b6ed8c65061",
    "sha256" + debug_suffix: "944db8580ef1e4e9c33c1924fa3b9029920b0c362235f0badf4453bdf2903cb6",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ad081e654e88dc3b8d65a5da5644f394d56461b6d2dfd79b07754e1836e9d1bb",
    "sha256" + debug_suffix: "8697b14889e053f0d82d33a64c70354a52dbd827dd206ea0afd2bb3fe377f543",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "382d5c7bdf2a1c6d50d47e22543ec9dfc447899c27c412eb83f0b73b2915c331",
    "sha256" + debug_suffix: "fbaf643804d8f581b79c3dd1f53258892aa353b67e170bd2bec965d9a7f8e631",
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
    "sha256": "546eff152cfac3699e20e1548259a9939cf7ac972b33698ff8a9c643a5a5f960",
    "sha256" + debug_suffix: "7bd6fbb38720a924d6d6c09a61383f5cc0b5336a1363943ea83373194c65d936",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "fdb888097f0c0d3be7895d483c779faeb27ac80623f75441e17dc3d147c3d260",
    "sha256" + debug_suffix: "39e5c816f09bda00ad8eec8025a305a5f01d92278d86151b2324ed8729cf8167",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "d4cba3e4151a930802d459cddb5004535db5f9653994fc8e88ffc18d04d892ad",
    "sha256" + debug_suffix: "99836487ff7641d6a2bf4f1b4b7c4189523d0eee0eb18d2deab5aa92e0f734f7",
  ],
  "kernels_optimized": [
    "sha256": "68533a71bdb652893cfeccdd6d605ac757d2122e476dc6d27163e982292f6461",
    "sha256" + debug_suffix: "393c04458decc5bb53f93e45fc0a25621fc5a1add5ee8f938a8ce0929b761bac",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "3d02304f54a8fb61ba87006f943343bac8a07486f2522fbf4a8d12219d6e9a83",
    "sha256" + debug_suffix: "559a67959546c4e0d7f91eee08401be169a35d9ecf079e0b3f0212ec92f5226d",
  ],
  "kernels_torchao": [
    "sha256": "6bacfdcc9d559856eee9d881bbd7405adf7c90445b384915edea2e864e2965a6",
    "sha256" + debug_suffix: "001ec6599be51fdbb9092edd4e8bc673f59ed0f452b8a42831b2c4cee2be0ed4",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "d9bc650f9122f23c826374ed939214b6a3463243d942f2a6c8ce2a5f1ede6df8",
    "sha256" + debug_suffix: "85bfeaa005d378ecafd40adc66dd39fb5b73fead25a26dc0986fd83e8f7b8cc0",
  ],
  "threadpool": [
    "sha256": "c0cffff3a90b0c6346ae7b6cc48c857ca2474dfc00a302e8e3d93106c63fe9a5",
    "sha256" + debug_suffix: "dfd87c93eb6010257a0167c4723d96228b4bfdc6cd0a856cbbcab9836c7da043",
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
