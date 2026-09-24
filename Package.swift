// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20260924"
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
    "sha256": "563c2f05c08f27a2007bc910436d9e325ae3a22c914f203f7bfcf17a97150931",
    "sha256" + debug_suffix: "d6831e6fa7a69935c9bd408f2b6ba5e52e6b65e262b7bf30068441e500b61e61",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "8711469a83a18cd704fe967f373baf615d75129c44ea3d88386ad3635749148a",
    "sha256" + debug_suffix: "dcf350df82277c331075e62b06e3347714081ed202eb359f17f85ab4d2406ac5",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "0dc974650b6d0f1ccad2190120ac1e8326317fb4f98da451d8241afb076635bd",
    "sha256" + debug_suffix: "e5b5e4b209a18be0e98cf88b6d66bffcbcc35b3ab0d259107fb3bbe8f6216ed6",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "b3d70c82feae8990b6e83ec253898f22b36fbd91cdea877a0358dc95325955d2",
    "sha256" + debug_suffix: "75ce71d1bc5dfdd1f664606b942e71b761c379fa075c3566b66e5a602a70e9ac",
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
    "sha256": "613d5f649a0c7fecc7cd49e518f729c38b5f346e43c19903dc56772b20ab799e",
    "sha256" + debug_suffix: "2ba2775804a6b7c4e7fa6043d9a9612dc06393d7fa2b12d5b8bd4ad929b22898",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "058b533daa2474cf8a3b1e32506610d96708ceb4be289a91bb5f47413fecf375",
    "sha256" + debug_suffix: "9bb17662d73307c5c9637a1f2cecaf76d520c444a8b487f904f636ff1aba6b1c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "63c8d16d82203b0eee6a51fb5d609b12b27564cc10d3a77e2550fca176253bb6",
    "sha256" + debug_suffix: "48072b0e64e0833ad982fe9e6d86a989857d08fa4f816bbff94798795cd069a3",
  ],
  "kernels_optimized": [
    "sha256": "67bc221af65542ace37843034d214aab0dc7ffc1dfffbc5218a13fe62656b18e",
    "sha256" + debug_suffix: "cbf8915672d1b276f7c8c5a68851c2bff7efebb7e13807e45699fe47c0a37316",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "c62c8175e4c99ca06a3a4de535358e409cd25d06f8c45d8ae13e63accf52d747",
    "sha256" + debug_suffix: "df33e8db059e414c07474c1d1f479b77c5f46cc6a2e8a52b6a08cbe5ad0b399b",
  ],
  "kernels_torchao": [
    "sha256": "63390c95096a369123ddcc29e87e7ecca80314f16a7f2758d41e83f3b4a9e978",
    "sha256" + debug_suffix: "3c38b813b032ba3853935b99a434f3573e386c6a1b3cc43ba53f7f2a6061a9f1",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "dc2d04e6bb12c9feac63c30639887f96319b70195aa8574affb621b69898d79e",
    "sha256" + debug_suffix: "fe15ab0b2f98a869e409aba7c879d2b242a50c4e245ae6d629d23b6fbcefcabd",
  ],
  "threadpool": [
    "sha256": "84a94ddf69106db9692c77a6a0467cf14b4434bfa58c58f6ed1bed300c3c2f2f",
    "sha256" + debug_suffix: "79a767b2d316f9a7d0ebbd883ab401e163cc48d63a3619438d4ee7655f01a554",
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
