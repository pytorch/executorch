// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261031"
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
    "sha256": "bd1bfea9ca3bf81516a360c08174655ec6e91324700498c3bd091e2bcb27c796",
    "sha256" + debug_suffix: "0261d2c883c3088afad5a1e35232efc59e2d1feb5cb9fbd5f9ecf8845676bbc3",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "d9441b2c42d8879c3caf669ee2020c20a888b7503f6f58e140452060119862fc",
    "sha256" + debug_suffix: "2bc26132bcc2b774239bf6ebc06d5d26702238f55e944da997afcd00df417dec",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7860f8bbe2b4349291a673a7b3b37c5684d4573a066dbc39b58e2f9ad989f3a1",
    "sha256" + debug_suffix: "225d72dc295cba12e0f7aba7e5209ecaf0aec43e9ebe89c5b1ee9d345a830023",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "5d244a2a391d0c6744a197d32f21203dfc3b6d9b4647334aa92eeec50f8f0981",
    "sha256" + debug_suffix: "31c21c0f12c9b51683d02e8b6efd67c916046716e4de428674a28dbf376c1262",
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
    "sha256": "a86f483d3fe9959dd8fb020eaea8c8ea1a7ac39ea9544b9c3b7f5878224881fa",
    "sha256" + debug_suffix: "c68bf835aeaf99a421ca8807b29b35acd084bd9c960607c2cf51e973ec9d9a9a",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "602e77c9059b5f64511b2d4f5f0c5f1538a8cfeb4f730849bb0a9b50810e9890",
    "sha256" + debug_suffix: "0c4eac79df0ddbc48616ab889f0ca90b162dfff1222231bdbd5a550fcf2733f5",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "1a647221977f31ae33c4e6bea690d7aa3ca97d46fddd8c9a000806061e9d1701",
    "sha256" + debug_suffix: "48a5a3d39e6d56e71c6a537fdfd4c983c1d8db4c82572484f427b47a96482618",
  ],
  "kernels_optimized": [
    "sha256": "2dd11958f4d1076462ff912b0183f207966301a9d0c3a8b3652c9f33d44bbe2e",
    "sha256" + debug_suffix: "dd7e1d3d81d870e55be7cd6318b23295d48e3ade23a140e4954a564ef22b5436",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "27820c80b7fe54384d77304f42d176d723c663777d09669c759b66f4c0842592",
    "sha256" + debug_suffix: "5ecb3d6e480984b85b121037c9a1d8ed2c73f558e29e5f2605b551627f4b9e06",
  ],
  "kernels_torchao": [
    "sha256": "43548bf96f0056387dc0450126a2754a36fb890daa522da577c7b5639e2c524d",
    "sha256" + debug_suffix: "e1a284e1a4c9e39fd5a62555b6c7e9bca18a1671901b78724146c87c81e631b3",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "72572de95e542304d5a3af14cddfeb1782113f07f82209369c1f6ff8b5e80c3a",
    "sha256" + debug_suffix: "c12f2889f64aadfafb033dc92d3f54040aab80defec22ce936091960e0803476",
  ],
  "threadpool": [
    "sha256": "9be698b8264d0396c94f602cf94c73e51169c18506d2dd37259b71591d7ae9c5",
    "sha256" + debug_suffix: "476af129a5c440e26adb68d45d2a679fe7927553c936dc606645a5d7e6e42746",
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
