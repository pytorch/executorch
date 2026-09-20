// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20260920"
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
    "sha256": "6ccd78292a76e6cfa768f4ef9ecb262dd4fd802130b71136b5bdfcfe4a4db3d1",
    "sha256" + debug_suffix: "cc18624d52a516950d8aa7775ad473eb4c935f113c51f579161012745f71432a",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "fe6c1cc29bc3e21de6bdd295f4d9c880a651f881b1f6ea917faf37b191570618",
    "sha256" + debug_suffix: "c8b88ec6d1118ae046d74b4f12f0e71c95fc6b7eaa5608e70bd86ccf899a1091",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "2159f7a1d7615952cf73a2233d881ec42968c1a185c37fff16edb25b4d10a7b4",
    "sha256" + debug_suffix: "af11235eb31560504f9934c519641fbbd72ef8089858d35e43841a4256a03ca9",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f20981ef8fcce17260d69f0fcf3c2183d5117718854d01179452530721448ce6",
    "sha256" + debug_suffix: "d56fd3a3b36b48b3f006081c9adab0cb480af25a3b9a4fc28f1fc10099ed1b41",
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
    "sha256": "17e4b54318150c31c380def57384a706f3282f2cf967637fdecb28a48c51c7f7",
    "sha256" + debug_suffix: "0ee7f47e2524b27090b14fc0a59bb2c91b38ec248eeb90f83f705db12f0f2387",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "18391acb2660d4cf74b0556dcac625e000edb79e1fd80c5a9f977ee3b7624d1a",
    "sha256" + debug_suffix: "f708d540dc39f106b1cdc86ed81d65586d1a4032faaafdeae82cafedd6551dcd",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "e370b0df3afaf03411a48c7af211602bf01614717661ffb8db6b18bdbfb717a3",
    "sha256" + debug_suffix: "f1302ffd72fe5b1d2da977e5a40ab6f0e3dfdca11a6bb97901bd41e9da83d711",
  ],
  "kernels_optimized": [
    "sha256": "150c6e429dc757a5086ea1c37e69491e58e42e17682840e3f9735051168b897b",
    "sha256" + debug_suffix: "2c5e9f2fab5a94fa66d578b21dd91d5562355111a1bfeec079148d182d55472e",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "1f3e086d1b12a3e6fd5bfadc185bc7fbcfac4d4b81698b244f2cd42842fc1337",
    "sha256" + debug_suffix: "9eb8a4a838610e3555e217f5a78a02ad7ec2d22e81ded4939ecccb7eb57488a1",
  ],
  "kernels_torchao": [
    "sha256": "203a1b45ce61580725981bd007fa2a90d4960ed37b4630281b123b8b990b0693",
    "sha256" + debug_suffix: "f5ab6cfe03a2c89fe87b2232208b45593c72257c4698d09f31cb9e38f37f4d36",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "d9514f584eacb32ea10d01f5af5ab1905e842a4e206aec251bac58217fd8c0b5",
    "sha256" + debug_suffix: "6aab8d2247d5727423e14bcd558816b902bd4ab03df0e2985c9c9a3ec9c80f8a",
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
