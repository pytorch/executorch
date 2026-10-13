// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261013"
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
    "sha256": "597a5485120c96fef24338d68c547826a05178ba842fd5275184f483ca155ff1",
    "sha256" + debug_suffix: "24ec3c2ee605890ba25b1f81389fa4bd239982bd0128cb265d505062808aa579",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "ddaec2c902818fea966400499b8e5ad9359ca086c777e0c7d31dd925aab6bdfb",
    "sha256" + debug_suffix: "ff5e478b0412d9bffdcaa0d45b485cd3f55f4d6883c7bcbd966064dca50fd1d8",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d8d9600fa3ef93d810d6a5a345150522d5b85d6d8f83b44d546057db2b1a3939",
    "sha256" + debug_suffix: "9521d7c154f83cc9c3cab838fcc726a5da51171a7000ee9e30ddf200122b0fae",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f857a0c7c7c5244a2c3f52bf2a5235ed741fdc9549af4ecce8ee3501c0a49789",
    "sha256" + debug_suffix: "f57efd3c2410761706032aa7479ac2a3f99bdf74b98452bc9a6e8d79d78ba5d8",
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
    "sha256": "36d7ebb8462f3fdc0087875682ed510893117b57c674c08bc76033fb7f5d5384",
    "sha256" + debug_suffix: "2793b5b6897873b83a8388b1a8ff55a480d536614a9e03314d1115b4fcd8cac8",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "4e166b4319797d32d84493b1fb6beeb8d2ec3694d0b3af2d06a6a0603758c8eb",
    "sha256" + debug_suffix: "32c4c32b353db541c601027b4d383f437d8b1bfcf5c7075beaafa14149479a69",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c1eac199b5d895f1ec4d8dcd0be61f36123682c727b465db8975e7c15a16d981",
    "sha256" + debug_suffix: "01c2b49b2bf678d646d5add8b4715ffab894f2cd2bedea830e95c1f27c89ac9a",
  ],
  "kernels_optimized": [
    "sha256": "23dac4bc5cbac7f5fbcbcb2d16ce71c787403d373aab975b5a27b90cd49b13c8",
    "sha256" + debug_suffix: "fdd42dc4a4bbdabb434f77e07773cee074699a923bf2536bcd7029f5e4cc2f28",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "a534fc636b2cfb0d7b01699cb028bbb2a01f8b403cef60406c6bbcf2c6ee8e23",
    "sha256" + debug_suffix: "16f5e83bdcb0b1b5517efc4f73b8c5ab05f0081502b6a0b24d1e7c2b85484469",
  ],
  "kernels_torchao": [
    "sha256": "287bbf2be1e9c8cc94128c50f25e098c8731a6a321441b486c70beaf42f274ce",
    "sha256" + debug_suffix: "c5ceb24dccd7ef2e15045947c9ddbf4e68cc643f3bbe13b80587585590bffe99",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "a821ea093e06783022f8052e17de8eafb0817cffb05f1b24aee7def0c0037c07",
    "sha256" + debug_suffix: "08694ab9e271a4c93e3f3540ff9d92144d254449bc5e74fdf0a288cc9b8b90ad",
  ],
  "threadpool": [
    "sha256": "61be33b8ba512bc467a30e81f5ad6f444af75cf06cd7a9d90f9c78e0caf42463",
    "sha256" + debug_suffix: "c9c4bf6b4f6bab572bc1d7b8580f616d1c60497b4a6848daf2939497024d0dff",
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
