// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20260929"
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
    "sha256": "c66c493edc441e92bd05316b1c6f94bcd83ef4553e3b22d95f87abe0eb1ef00c",
    "sha256" + debug_suffix: "f9b9c140b265ba4d6ad3d08941e7fbe40b5ee36ea23817213b886e9e9cff31e4",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "366ac380eae4761104427882537217133506c175f8f284c19ebcb8957750a5de",
    "sha256" + debug_suffix: "ee562ae152db5d9b24ba3204e3fb9228b3c553e193188b95818d616cbe25cec3",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3236e35b7bc3a9a38513f781d41bd2fc670c8bd72e648d2b10ed70776e832fcc",
    "sha256" + debug_suffix: "a4e89fb85d5246e98a64664dd579454006f1bfe60ae52f3acd87afd0a41519dd",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "ba11c5d7a59dc5c5fec285479f990ed9f7c335469573078522a5c3882b41f285",
    "sha256" + debug_suffix: "77c4b5d423197863fb18808db86d409be4fbc5163bf41e4fab02640e1bdab618",
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
    "sha256": "5643192976e6461c6092f866583e592cbb5b36d13560e920d4e58f55994c491c",
    "sha256" + debug_suffix: "daa64ae90296136496ae02ba256b0c2f8edd1d6423e3a5f4d1dceccd6aff2c1f",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "58313e9053f70a3a9856ceaffc371875d170f8e5180ac713b9e47729beb4ff77",
    "sha256" + debug_suffix: "6260629943a71e89dfc582cfe8c88c11ab8d8d105c5d82c782aa39afac6cb397",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "e30cbcb5391e9f125ccf14d90b8758b8cc01eca404e244bd0ed96040760ac249",
    "sha256" + debug_suffix: "4942a015693132fabdd48e307b43593267f6ea6fe320af0b47e80902c4f82095",
  ],
  "kernels_optimized": [
    "sha256": "afb2a25f75f3eac9f596ddb627df3a29001bbcdd93a4095ecc7c284afa2d92e3",
    "sha256" + debug_suffix: "ca31617aa1f77817ecef45cc96dad5d950491d6502a8c86b3c1a2b35eca72de1",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "259a5f52cf245c53bf8584fdb4a56111861bc56d9fd4a92110798d41b5e4b265",
    "sha256" + debug_suffix: "1289d37540efcf138a54bca48d63a8a19fb1ea59c68ea500a9e91c6b1844a2dc",
  ],
  "kernels_torchao": [
    "sha256": "98140a15ef0b3c12bdfeb84a8a3dbf3cc4285862bb7d96ecb012d9ce447ae48e",
    "sha256" + debug_suffix: "140d6aeb953a1224dfca3af5d6f0b214ba063937e92cc5e90029f4354b37d180",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "b1c5c8ebd54b1ed282cf00c2ab0ea5675afe4e50f915ea6e473a10531966076a",
    "sha256" + debug_suffix: "a9093826be1e8294e0afbd32dad5110c39f878524c82bfb56c370ba69753cb8e",
  ],
  "threadpool": [
    "sha256": "0a069ea58fde7d798d883299a0cfa3751781ff2d83c94b945587a33bed9052e9",
    "sha256" + debug_suffix: "4758a1ec42acf05c888d9e3d22462cc777ce2e97ad1bf8f951c127eb2e24a9cc",
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
