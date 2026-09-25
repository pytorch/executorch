// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20260925"
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
    "sha256": "4a6cc4bccc55642a7b1e7483f2b630b04d50a0e352443f1936a1eba10bc4d0f4",
    "sha256" + debug_suffix: "3fdfee3113d78e33041d5d2ec5e4259ae195f055245e2014de1d55a82c684975",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "1c039c0ab9c0388b13defa4b8a1a9d0283a17342c68f7a5b8b718f019065b677",
    "sha256" + debug_suffix: "953ab70cc97c0d114ce7ce1ea96011a7b6846951dc72b8bc186b7f04bd4b1a01",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ae926b67496075eab207b9c8001bbe6f6803fdd8c5f4d97c5743ed2efb1a606f",
    "sha256" + debug_suffix: "cda012ff8b4d0b1c8a8bc05743efe45ed9c51eb296a16a5ef591a98c838b0d33",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "ee50993aeefdd184b3c1d78f08c7e9523c60e1b26dd6f0b771ec5a2a9d151fe7",
    "sha256" + debug_suffix: "7921bd33ac0f8c43348e6c979b47cf025a0ddc453d51f208e31c7dd55f746420",
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
    "sha256": "27ebbf617f82f57c7899df3c8a5430e550f03b2311f7e5c8bab35181c1c67c52",
    "sha256" + debug_suffix: "70a90ce7a71889fd9f665f3763d4c50305f5cb61229e125c67a853e1e79406a6",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "a637b3438ea3a89d7d6aa291ad15bf12373872437d927c1d8da9592e5ca25030",
    "sha256" + debug_suffix: "20319c7b35a3262a28e67094c0931ffe763b4b06a3dac76639a9756e3054e8f1",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "cf60817080de57dcbef797212c7bbf80eb1643e4285d09502ca4d29782069c6b",
    "sha256" + debug_suffix: "6a5f2c6445cf5c3e89f2f3ea290e8d17d17bd4000fa11f6e1dfba70c9efd2034",
  ],
  "kernels_optimized": [
    "sha256": "6a3833b6fb98545800182d767c259b1b401415b8fd0c745d3914fa1d99edcd6d",
    "sha256" + debug_suffix: "c45dae2949a7b96524fe27df5f76bc29f664ca4e324d512bbc9ca52ab373f3bd",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "9791811f01cb99bdd7e045f49123965ec1f9e3abc072d9b1a2b4e81b4a9287ef",
    "sha256" + debug_suffix: "c4ac397fe6bb74b9c055a0b514cf98d6de80e8f7a5eb0dbd159180300277e591",
  ],
  "kernels_torchao": [
    "sha256": "4e6c7a709f27d8f304aa85951fefd82eb34e60a21366706d346b50977c879d09",
    "sha256" + debug_suffix: "615ca7286a837358b1a37279f48ad050c487a19f277577416fc37ffcd93bb8ad",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "f4f087d5c4a3240968d92203ee629a055d5b79e2bc59a9b5bea34db183590175",
    "sha256" + debug_suffix: "8ef93e02ca9daab148ffbd44c4e31bf4bfb4eef6624e90f334da0bf6256f6935",
  ],
  "threadpool": [
    "sha256": "8c61127ea077f6ae80a391ad048475ee0b83913c373bac64350c4d5748a46bc2",
    "sha256" + debug_suffix: "6f789988717ab6780fceea0a4c79cdc247a9878374c7cab395b8ee5c63d9df97",
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
