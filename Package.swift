// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260906"
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
    "sha256": "533cfa6c66f853f1af450f026cc383f11cf53b426e6f7457c1bcfe0ab8adac51",
    "sha256" + debug_suffix: "4a2c488251fb82ac3c0734a4a13626dce7153e69db16f30753cf4ab64d0c1f1d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "d211db524dafc76b4ecff424414f9c177be1078776950a84c6abc2ec182d70bf",
    "sha256" + debug_suffix: "bc2921f7878a44218733d43e5ef2ed1baee0121ff85bc1df69727d3d5a8a61b5",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "98f7a80e497965e8eb26d567f70a101957304527d8df3b319b1369dd90137e1c",
    "sha256" + debug_suffix: "3e5acf56e86a1cd063c4a5efd79b7cd609ad37c20c6357d235b07c50726455f2",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e513cee65cc18cb2fcc5fd61008ecbc1ebf8a76408a5b3db94c2f23dc679c5e2",
    "sha256" + debug_suffix: "f4ddcd83c141a9028f4073c453214ced07e51cfd8ba6ab1a7bccca61dc46175b",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_dump": [
    "sha256": "3597ba3dd505d85e8f4daec636b450efb4b69db0b4ac132a8402705d60b54d51",
    "sha256" + debug_suffix: "57fbaa479c9c1b3997d630478c5e38d1b09431bac6d7f92ea0efed53af871556",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "3292ffc59c5f45f3ac3b6f742729e9278491234082b00fe69175b4b088de21aa",
    "sha256" + debug_suffix: "6c7ec4271adf4a4938a4148bbc21ebb930d5cd8833125ae546ffc14789e8c816",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "345745f24a213c545eb9a98557b228596d74772ed1342cfab989f0faa37e7164",
    "sha256" + debug_suffix: "cf15ce871a8369914d4a0d6712c0e6796542d80e8df87e16b8f108b5dbe95d42",
  ],
  "kernels_optimized": [
    "sha256": "155027849cc1519df142d40b57ba28130cba99cf48d15d5bda7e7cce9462e0bb",
    "sha256" + debug_suffix: "17f728c316fad21f8f1757b7dc075a7d45020c35ced09268f0095ebfbcc57b94",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "d064b99b08f878d7e18945e6ef97ee417f743e38bf9aa3c828240d3ef70114e6",
    "sha256" + debug_suffix: "9936afc8a7859806879f7801e66f75349735d295b13fbcc574c09327593a39dd",
  ],
  "kernels_torchao": [
    "sha256": "f2a294841c7e07dc85b30941dd446fc2e915dffb4f12ee914bc381ec3e771fd1",
    "sha256" + debug_suffix: "7db7d0e2e430fa747d415ee2d2a0bda6c9fdd0cbabc27ad97e0d7709ae5c47bd",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "1e738b9887be0afe25e92f310fb7dfbea473a218d983661b48ff2939fabe69be",
    "sha256" + debug_suffix: "042de5816c0d3c5de88044bee15e717f094c549608b83d49f28ba4a78f3d1b15",
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
