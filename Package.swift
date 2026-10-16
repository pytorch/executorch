// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261016"
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
    "sha256": "e08a1cd048a9e4f23a8b5b494a3237b0e2cbb8e5e3219d4ef1fc663b6569f548",
    "sha256" + debug_suffix: "a6e65a0f893fd3999ab2d6884cc7ff4335692c9f7cc9d5cb2361b3a5c6118afa",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "d59ffdef2c9db4d305daa9cbf5c3b6838f481fdac89d8475f5f9f2674f5de3d7",
    "sha256" + debug_suffix: "fc6ef4ca3ce71c26e33922c9af34450eeba6029820bf14310b123e815a742210",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "c7bb5e4e3ae12536047c07db120f9928e7856d0ae0b0163e00231f478921fa3b",
    "sha256" + debug_suffix: "9679a52732cc3625320da4d9b23fe22778fafa47dfbc860e5c22f68e5e5ca539",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d5fcf2e76cd5a95421c864c214d7eb2b5830435f49ef2d644f4d9c71468f0e94",
    "sha256" + debug_suffix: "59d058aa935a0e9cb18bff90b48fb6073626c333a30a20e3cc2187cae4816cbd",
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
    "sha256": "f38ccf0c8a51cd47b1b847c5bb227b7addec5e234273b18550ce6d37d7dfc186",
    "sha256" + debug_suffix: "04cdc3be9f28a68259bdd18d87b59e75b62a837622ebc997c25f4e62e2e09e94",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "fe324bd5335bd6962f63f49c2a3dcebb644a7a6df678a8f7d2deff870ecb511c",
    "sha256" + debug_suffix: "1d15ef84b84bebc37717a7c00fedf3a714869ecd868185ea7b6d2ff85ab4df17",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "4efdfd6aefbdf5bf976d52a7e1b0924d887b8c21b83738b95c4532e7e962200f",
    "sha256" + debug_suffix: "7e38a8e3136b4ed6f61772c7b634b2714b01f766dddc1312be6b24c0bdbd1ca1",
  ],
  "kernels_optimized": [
    "sha256": "b92478a5bbe30c48b6e471508c7d382b132e1242cfe7445bd8c7c76eb18ce6d7",
    "sha256" + debug_suffix: "b5887f8e6c3c571d1406ed4c3efe10e1d72a6f4a034076c0885bc30971bb93ed",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "0d02cb3a938db89748cd20ce7badfbe3a70db5fa3bb28d843f0249ad0cb2c90d",
    "sha256" + debug_suffix: "eca1355cef96241de4eac417ea91202aef8ae3aaddcd2a0cc5961032a5c935ee",
  ],
  "kernels_torchao": [
    "sha256": "786a38e27014c46592b918d46b27b42e5e93aefc8c6f191185ad5874f5cadfa8",
    "sha256" + debug_suffix: "0fe0dca82fc9d711e44b94b421cd693456ddf1e2224102ca2e1f100bff84cdbc",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "5d5e45fde23d199e888387493a7cf46036e058392016e33f9d66a96e1964eb1c",
    "sha256" + debug_suffix: "9d044c3e7e8cc9112726f1465ee0ca25b254ddd006991fbfaaadc4a1443e5042",
  ],
  "threadpool": [
    "sha256": "305dcbbb630b67cb0ee05c83e8a773eced176559c76927ca9db9d89f6ec7a3d9",
    "sha256" + debug_suffix: "b72b4594db60f67a194e9b356136bf6373b1f015ca9d5b18a239ed0b84759cc4",
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
