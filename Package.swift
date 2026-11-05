// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261105"
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
    "sha256": "87a9b445749ca1a9f79978b559468421445a47e4396d428c281513cd4f2da1b7",
    "sha256" + debug_suffix: "f3e563cd2ab6cc4ea2093f78435177800248604f874d8406913bb926039eedfb",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "8388d2bbf1171088ec1c7b0fdf5072466757867128211f2f420b2d245d6b5f6b",
    "sha256" + debug_suffix: "409a1da654427e8d5a2ee3a2b839e87403ac99612aef20e8d7bac664921708d9",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "02fc84e115713df3c8ff017ff2f5330afe491934fad93109d643dce2400d1f7a",
    "sha256" + debug_suffix: "6b4ed457b165bdd2729953ee457370f64b341d637e8ed977e7edb96f76cccd0e",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "67d349422684124743bca11e8206883640cece8b3778766950ae54b9d5d42922",
    "sha256" + debug_suffix: "be07251256d8aa0d0dafed793043025303dd23342503efb0cb9c2b67a940c9f9",
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
    "sha256": "a95d9f589b33e06cff7b9b6a8bf8d5f09ca6117d86bed4e20d5a945b84281a14",
    "sha256" + debug_suffix: "42496a6459c7bd5863533582301ae5ed1f25be5098bf8037a1b6954b5432b2b7",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "78485111832e08b8d88c09630c0e205db99c60c7e554361a4c7b92888d7a2c41",
    "sha256" + debug_suffix: "cf42b9974bfac4a09ff4c34edbbefe959eec5bbdf730e659a125bb5d8484d764",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "0df97669d09293dc9633cf5d1679d79b30275f3baa7785c9b988fa6f6ae47f3c",
    "sha256" + debug_suffix: "937f283f05a905ebad5925ee17691bd8f96d77dbfe28b2210f210c4ba0601f99",
  ],
  "kernels_optimized": [
    "sha256": "7e50405b87aac12265fb3b192d17f75bf3325f8c8adfe18b6085b1c4f0cebfc2",
    "sha256" + debug_suffix: "1539b040e806dc905faf860ef380886fc91d023686be45eebc7a6d9d6f0d8f4b",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ea5394839c0d31045b4239bde1d637051b2596c66867af21ef05b26f49af6d44",
    "sha256" + debug_suffix: "49dc2182ddd25c038b1f98182fc4d4407c5235ee61521dde747e2c6fc78eb323",
  ],
  "kernels_torchao": [
    "sha256": "34b27bd02e5424fa4c8b73d9afa7bb083558f22a9d260072dd9b5a47e8866a75",
    "sha256" + debug_suffix: "e7bef1e97ff8083064099b829d13514fa51696537eb415c6a98e2892f8c1a2ce",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "46964486130c014583daa71be2b40168f3c652e217dba16ced52ff1d7d5adec9",
    "sha256" + debug_suffix: "a1700bebb8bc8590ad5c5f9f6120f5e937610df89882a12fe6a117f8463270f1",
  ],
  "threadpool": [
    "sha256": "d38e9278dbb48e9a5bafb59cbe2cf27e804488c7bdf7489e3b029e0f0d578c6c",
    "sha256" + debug_suffix: "9d2623774f1acd10d3fc3af66b679a22f671b2777f6ad25244f2f9405e11d0ed",
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
