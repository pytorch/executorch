// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20260926"
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
    "sha256": "f522bac979a4b21d862180a0d3d135b5a0ddbef310175f6e0121ae7476315989",
    "sha256" + debug_suffix: "d5b4799ce1a6a4ff3214af8e1d855bfbc0fbeb2d4e2cbbc59dac86dc81bfc227",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "4ca8a982ec108543432cdc6baced4f62c6704e2075367d1575100fa50533a0ff",
    "sha256" + debug_suffix: "5c06fd26fb0eca8ecfa14e4c62b7cba9c472d4f81cc02d44b97f841931f01151",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "88b92c900ecb3b22f93ee2d12a94a1026c17664b4a13def8f63222a168dd2f24",
    "sha256" + debug_suffix: "7519dcfc326d5af34640c6c91c8b987b4a16f4a36c4b07941604b38eb9026a80",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "d9aa2d4b3e7d212eec8ea9f4abdfa81e584dd6e627368e1585df6ba6887318f0",
    "sha256" + debug_suffix: "71dbb686c7e9942310ed63b1353b678c0470242b20f78c4ccf9626bda7b0e5da",
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
    "sha256": "1720b524ef6d02f08c1e2015d5bc9f58151f3caced2cb28f89e8b2c1fffb5c05",
    "sha256" + debug_suffix: "2778add7a372d31728db36bc041bc63e1848f8022e01bf714d74afc0dab47637",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "520826b7b7837ed021508cba2a510363286c46a23b09f6c116cbbdf62308ad83",
    "sha256" + debug_suffix: "09681d138d0a817085f46261b11fa390b482064acc4986a62f3d39df4a35bf7f",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "3920d6c31b7cca378a37c7c00e46863a062d9ef976ea3d547cd2acfd51d6f1b3",
    "sha256" + debug_suffix: "6c149b645d564d395541c3fe54487de1a4651f8f893844ce41a9b1575258a991",
  ],
  "kernels_optimized": [
    "sha256": "967c4893863a345a3156ac79c919ebea9a26bf4d0b8a367c8be701733aeaf7a6",
    "sha256" + debug_suffix: "2a6e4b83651b9bc0ed9138b9ef16c7c38d7086d7f0ee16b2432db403f28e52d7",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "d9361d5ffdf75d0879b6d63f9a96d195cfbcc3846f1160d912755f37dfb2f6f6",
    "sha256" + debug_suffix: "69b9247e23d041414fa60f631e8d341cbb859dac27d16bcef416935d9c2fffdb",
  ],
  "kernels_torchao": [
    "sha256": "6ac38651499372485dbb3822d3bb30a3f8bb4eaed2b9b5a86a63841f3967f064",
    "sha256" + debug_suffix: "5839a8bf37c345683f4409e80bfeec9b9156d64697669d1f01761c42a08d30e4",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "d0634d596689b894b676cb3f3c0f41483bb7d10fb7291c33897c2067a06b0bfb",
    "sha256" + debug_suffix: "31e4a66d481ad22b199756c17445e2b775858e4942bbe0384d5dae9a0ac3062a",
  ],
  "threadpool": [
    "sha256": "e2329d0db38ec0eba5c146c58e4268ddf74343096bcc39cb0e5fc747bc29e82b",
    "sha256" + debug_suffix: "1e06efd779537119e7988ca82724679255e1123b4a4ae1af7786da84170854bf",
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
