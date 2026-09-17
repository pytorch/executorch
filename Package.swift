// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20260917"
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
    "sha256": "35aafb61f951ed5b796c80122f6b87276b2c33071301f25cdda9f95e15e4be37",
    "sha256" + debug_suffix: "acdb9770297e7ec74b5bb1ff6b3177f954a75f1b0048353517a6e255f6cc5a24",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "680b0c030b47ca552ed886df05728d90e318c71fa6f38f1b57eab2d87be03022",
    "sha256" + debug_suffix: "27b9c684f8c2a7d8f9c9da3d9c881d87f4d787211cbcd4a8d7763554823f0dc9",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a3b61a7e1b275d17bf36b98bcce6efcfb573bdd3d1bf70add86c6898b610ff9f",
    "sha256" + debug_suffix: "7c25f82fb4707b21b5781d55478d8d68d7b9545fec3c1f45ec5edaed5bb43175",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "6db85a8fb4ae3938d4003c90091c3dac9df7acc5b00594483db315109de83458",
    "sha256" + debug_suffix: "e9eea653b7055156171fc91ce4b8726bbda4c9c382cb938d6a8291f35d47d175",
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
    "sha256": "a2170b569a6bc8bc6b9fc3f8b4c66e2f06522a6773292e0f97d58de4557db045",
    "sha256" + debug_suffix: "0218552da6a56df9af1a681fa001e2fb7539e6ef130302ccc46bed00a2ba7147",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "ad5e20d289821522cba0fd2cca259c2a22bd30422a964be85d4430d4cf172308",
    "sha256" + debug_suffix: "3f06e0e78dd004e5fe0cf1548a87244662238ad7ddde8f9f901d82f1f4b4a4bd",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "8e19fdfa1fea63a522544055d297ff8438682bbce1d5b4843abba85aaa1ea2da",
    "sha256" + debug_suffix: "42a41290ec8f24a694119d5d47a1d9d870b92829f8398ce723cb9dc3f75475f2",
  ],
  "kernels_optimized": [
    "sha256": "5e6ad0c16f5121b12995d45a33a3bf8cdeeb69ba32e8c586838c7efc17016c1e",
    "sha256" + debug_suffix: "c1bc57ffdc25976b73ebc80d4edbe2054c8f5106411672a7635873051e45b820",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "13df5f2b930d87452ff7d7e10c3e88e140cea9bd6495c17b0a758abb5f199166",
    "sha256" + debug_suffix: "03d7ca8579b26cfab3ca8adb085b7c6844a8726bcb531e7a15c4c4a82af0ac41",
  ],
  "kernels_torchao": [
    "sha256": "103c2847745e8e80ea2238ad83bda96e5fb0c07e0139c0809c66fd288a3a633e",
    "sha256" + debug_suffix: "156580c5db21a804230975c7ace8aa17a83d673b6b9ff6c6bffd040eeb979838",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "a8c235204595b809d7934695cc9b7b8b5471f793ce1aafbc25a223e1c7ecdaf3",
    "sha256" + debug_suffix: "0429cbf0a09ac82b4fc963b81946696bbc273b6c63aec9f5b91e25bae39b4f7c",
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
