// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261125"
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
    "sha256": "a0824bacc619472e435a200ff5402d182a9531b6918081e1d2419ba65ac37efd",
    "sha256" + debug_suffix: "fc47d9b46df2cdbea2a4be79d9620d020dca4ec9620c996ad2558b0107e1eff7",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "77552a38cf8075927782cd2170d1d7eedace20059605b4fb113c52056e90a1c7",
    "sha256" + debug_suffix: "e0768521d47b53e4e55c8cdc4bd59e1e79e1414f68da49c4c63548c5f3911991",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "3e37a898083f0dbab7fa21b7ef03228622abb070e7625042132b5f2a03d7c518",
    "sha256" + debug_suffix: "58862fed60ec9211ffbcc702c00b10112dc98466cffc1ff0400977f403c5ac14",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f5601b6de58ecff1f34094165e1579b7b246479b7fb91f1a988dee5e01284e51",
    "sha256" + debug_suffix: "4d927cf352c535c1513c4f0647fd4be35fabb4d071943af935e555af00acadab",
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
    "sha256": "17e1effcdd5919eed920c7da2d50c18a359be0be4df6ea14e201e89ef771bca2",
    "sha256" + debug_suffix: "cb0576d74528789539d17f94b68dc79c8a1d275dd0678ba409db814f0f97c6a2",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "67b52c0447b6844169ff75d2b8a22a9d3ed11f89691f107722738ed625f8f5ec",
    "sha256" + debug_suffix: "c481d59b6e2bbeba1ca5e8672a05bb5c4d5c1e0b9c7df757e231dd73ca0d883c",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "6fe45327596e51228c855a6aea9db0595aa502d5e8bdf45f4842a96ae6b181cd",
    "sha256" + debug_suffix: "eec81ce87fd03964f11b107c345e4c32004611a303799c73af54f17c97970fc8",
  ],
  "kernels_optimized": [
    "sha256": "7ad970dcff34c6b33c08ee7d0d98b976e7d8552a5e738256f74c5209623c38ba",
    "sha256" + debug_suffix: "b9ed1cd33aa6d8fd168f851afc3bcb32aa92a7956ab9ced1f28bfa38dba3f0c3",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "8809fa243268de89c1bb4c43169c2883365633631d00fb5e51998d5480e0073a",
    "sha256" + debug_suffix: "380eda21a6db72226500cf9d85d15d0fcfe057f85ad9293c878affb108ccd637",
  ],
  "kernels_torchao": [
    "sha256": "ba0cd91c162a5653ea1e7415f0f73df74a2105a23f2ebd350c72a97260d9070e",
    "sha256" + debug_suffix: "c5cfd4d8617935051ad44e095a75d4e2d2db1d71b463ec814e14a82e946b10e2",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "135c82fdf024f314a0333df04406f896f822c5f2cfe4299402810cc9f8eedf10",
    "sha256" + debug_suffix: "15b725e927cd622494b8b2a0dfefdc74228fa218032d1d96b0aa203939ce17d7",
  ],
  "threadpool": [
    "sha256": "d6a5144bbf502abad580293b89c23178e1fd0f0028b9d5a376859e13e9b190fb",
    "sha256" + debug_suffix: "c04f5e57e8e4d84e7c3d77bd3c2ac4483165ca91ccc7cdc322c2990858ea9c65",
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
