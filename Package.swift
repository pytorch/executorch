// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261020"
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
    "sha256": "9827caed4c648badc8e1f8398ebd4ce46dce1054f6df8d0604b4e689e92586f7",
    "sha256" + debug_suffix: "89ed8cd2ed58cf15e84161896a39bbc52d7cb166c60c14b5a47eb61d52d8460f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "3bc0bc31b7c9fd2104aa3346a51416a1d7e5986f004d8a6e0745cced00933046",
    "sha256" + debug_suffix: "fa9140728ffd06574357c1017a642b5650f272f5131c222879849726cf544211",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6c1887afecb276f80a7b7890cdc286352efc67b1c376cfb53f9d61d251c79de7",
    "sha256" + debug_suffix: "4a3f9f6130f28f1356b18445c4ffb48685ff100ac7367fdea7dc8c998cf48ff2",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "75648a6cc8eb1072513e21c5900eccc7f5bedb388a33009bd0f8d128d2ac49b4",
    "sha256" + debug_suffix: "5e6af6360d010f395295f61c3e6a712236a190b907cf0afa3e26cd4a40f635aa",
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
    "sha256": "9b1b3a279e2c9b6aabb3f682af8dd62e05abf973d1582f2c31968b6947d811e3",
    "sha256" + debug_suffix: "7afa781087ab1513bd40a5655062cf58cdb78f6212401ea89fac840a2e0b98c1",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "766bc2888a708512f30bf41f89e788b500c2c79bf4d0c249799803774608095e",
    "sha256" + debug_suffix: "a843b36f4ad15861b067d17fe2e84baa3634516114c1cf7b043bd7580014ddcc",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "07c93cd5339b8b51ffe1c5f3aa5b52140f3089f0d42167c3e6071396ce17e889",
    "sha256" + debug_suffix: "37eab334bbe38287ff1c7736d0b7fc18129cd004cc2cc52695465d5136abd0ce",
  ],
  "kernels_optimized": [
    "sha256": "08693f0bdf64253a68d583a4d06535db420e0393fe08328b0209abf9d23124fb",
    "sha256" + debug_suffix: "51185d3d22188f81d4fd6069586d58e2f52ee39fabfd9d848e3c8c770a0db379",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "7452844f9683f0c475c23a9eb8053b706e51e92b14473aa9c4a93486bc263af3",
    "sha256" + debug_suffix: "cd753a17a3aa71f4769d8855ffe45571a06185f9af94fc1554b9030a5d89d3de",
  ],
  "kernels_torchao": [
    "sha256": "e2ea97f8d06a0bec856a8e4aef32cca263af9a8f1438000b81749b6364143def",
    "sha256" + debug_suffix: "94913a4cb9b802e67d00a189bbc8845cd0f135c93590ef912d761af8e56b2eaf",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "290cf1296e5cb3924b063005246835aae5a67445f608736c7c3d9dde093dc245",
    "sha256" + debug_suffix: "b8f945d2808f6241b52ce5b1421645338e0c8e0f727f8358f16c75843b666a05",
  ],
  "threadpool": [
    "sha256": "ccec412fc28c08c6efce3244a7ddf077adaec399821a2679363133a64de3ccea",
    "sha256" + debug_suffix: "5c8e6a91395068405c66dcf212a27eae0e02dab36f6f5ff51d99bd3a4dc449f6",
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
