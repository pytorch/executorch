// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261116"
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
    "sha256": "2517bef8f62e998190ec75f92b4f59c9ee8938e9c3755015f55a9f0c343747c2",
    "sha256" + debug_suffix: "2223e283de0b6f5fb25b3fa8071a6abae046d5e9337408c580ecc5ac7a0806a3",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "fef3263f3365ef2df910c4e8ff80b87389a621884d5181a0b3b9233e33502a91",
    "sha256" + debug_suffix: "2cf793e0f61959346910d76bf92676683ca8aac8068ea0d495e2a8ead246f7a5",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "f322f0a1d9f307356eae44984e5d578eefbc4772485c9998578a2337f96b8991",
    "sha256" + debug_suffix: "e62c5f74d7561014b66a7945a43bf75c31e98f9c2b8b2a8531ad49985bf9bce2",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "66b2b3308915595edcfd5628afecb6937e5dff0101cabbd2ed5c9af97a47d656",
    "sha256" + debug_suffix: "5af9afa1eb5bc56842a3d7c695f0bce270015c204bf9945981da9da9f5b4bd9c",
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
    "sha256": "0500a7045a95c0113d02737499b32c52da94793ecef08596b999629ffe985aea",
    "sha256" + debug_suffix: "1a673ffaf30b2ae6ac604b20077974ac414d7669c841afd7491904b83d1063a9",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "f20eb122748cf25b036a973983456757ab324d96548a1b96866df5a57a723f1e",
    "sha256" + debug_suffix: "3920d54c7b7d27d222d33b6721220371c2384a9230f9dbcce9ece0ead2114836",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "53307d94af57aa90c252fefec4ff9098dd279d3bfb5218194262c3d803ef866b",
    "sha256" + debug_suffix: "9b103101fd29dbac302f57100e14cad21d331544cda3dc491cc858c287b9b99a",
  ],
  "kernels_optimized": [
    "sha256": "a3d929bed5cd9336c4aee53bb91adfddf02d0898583b01582e228464cd97b1ff",
    "sha256" + debug_suffix: "e1668193899a866150234d8889b5ae5c64f032ba3621636cc33cbc817ed7a010",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "c200862e95d4eafc713468fb849aa6209be681ccc76b58c9d3ad7799ae01a449",
    "sha256" + debug_suffix: "178f8ee068704a0dd08837e7d65934725a09b9461e9c514ea4c1fccf5d94fdf4",
  ],
  "kernels_torchao": [
    "sha256": "9b9c8255df11b65e8b7905886e206176fa9b3db477ad9570f943cdc3db3ca0e7",
    "sha256" + debug_suffix: "04bdf6e8007deb3f5e1aeac59da047e2463279f72e0d98d3c656273b1d70df8d",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "d27378939c3afc68f4a75818af3377af2545146810938e242f6c3782d3e9e123",
    "sha256" + debug_suffix: "dccd81c3e1f1a4beb07694490b964a3485d47e57725284336033768f2e30a92c",
  ],
  "threadpool": [
    "sha256": "4c66b8e1cdc4f20024f9910a2f2a1285dfbe460902c2c5b96f907bf31444bd65",
    "sha256" + debug_suffix: "c61798475e8127803f6476dd6bc223495e5551015287f598175246123f7ee95c",
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
