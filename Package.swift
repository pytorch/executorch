// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261114"
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
    "sha256": "7b73e85a782ca03aed211f89fe5004868f7e31067aba835df2a398735e06fcd1",
    "sha256" + debug_suffix: "658d05d3db5112905937e6e582cf5ef8b952dffbb7f27cc3cbfa8ea5f949409e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "7acd6e1e8206cc1e3aa762c5137b831c81fe2c761d0759c1c67dc1234cb4226a",
    "sha256" + debug_suffix: "bae560c7821c08018b42de60d9ab53eedd20cf1e8627b71dad198ebf7190e9f9",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "92cf0a4749784ab9d073c807fee20357ef049679a076befe19a551affa6597e0",
    "sha256" + debug_suffix: "4d28aa93642109285ea04f1c5e8c359e70174bedf49d68a4deb72b613fc5bea6",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "1ccc3bc43846b294a98346294b2906cda41b9c4b2dbfd290ff2caae5dc79becf",
    "sha256" + debug_suffix: "55ee6463881dc593637f598a489a958e361b391be3df81ee8718f71e50a33974",
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
    "sha256": "878b0f7e347267138c5c1989c793af3540abc8e5b5ccd30f6b37c076ead5b848",
    "sha256" + debug_suffix: "733b54c47570ed269453e6cfae2dc5ec3f96102899a6e56f4606f6fc62481736",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "cc452b75c82f2ff35404f1cca76a4159658a6d8b31245f222fa6bda9fffbcfc9",
    "sha256" + debug_suffix: "86c75df8f9efce9825ad4b8d06a31411eb2ab760db6c3a03d500c6a356cb271d",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "2851c45c6787d4ffc3b8d96aaa877ef081bcab886c24c5935f101eddb9b448ba",
    "sha256" + debug_suffix: "69e7ee12a97fd19e51cf8c6dee1bf74e725097de6f4ea7756a8f6f24c43690fe",
  ],
  "kernels_optimized": [
    "sha256": "29cfcf7c3262ab0b825e60a86f8000cfc986d3b35a3a4e6abe6371a0ab5d0bba",
    "sha256" + debug_suffix: "a519ec90dc468e11d6fb32717bd461092ece4764e544ea52b4aac7cf338d233d",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "649951621a54f1faebee394e1707e0becacdca3fd96d4e8603b496f148e008f8",
    "sha256" + debug_suffix: "895a33aa6410776842c946900c439e965f3884682b34b222c63c8536a54d4b9c",
  ],
  "kernels_torchao": [
    "sha256": "37ace189fd0400af5a4810497fe04b75a645885d4eda96a3a1aeb085c9e227d9",
    "sha256" + debug_suffix: "06db85d8afb512c4f8417407d867a47e61374dfd5c9a825e080cb3a8f138be9d",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "a7d0cb6549b7fd7c2a7ff947ec3e95629115c3a5818c79ba9958eef858c80b8d",
    "sha256" + debug_suffix: "8887c013d06b859b0c2baf80011d78e71e97d57cffeb84f36146dd469989ce61",
  ],
  "threadpool": [
    "sha256": "6f1acb737d7375538630200be83a22c3462ef8ad3e79f949063ad5c6d5449f71",
    "sha256" + debug_suffix: "0ba6176013191afd166f25b165c105e7c1ff2634a11e7b277413c0bade53bad5",
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
