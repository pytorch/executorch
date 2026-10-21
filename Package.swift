// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261021"
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
    "sha256": "08ea91764d346f970705f7de512f821354681433036706bdc95e4b467373cd4b",
    "sha256" + debug_suffix: "caa71db88242f447ac2c20146efb3356cbca333134afb2e60afaeda2f097e055",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "363a223c03e2654b7da006118591f4b8dc31b00bea42d3958008a0ecf87f33ab",
    "sha256" + debug_suffix: "8f83d938fd5817a526af58811af669e68263a03b0553e4ca7fa5e7b93dc38f2a",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "938c4bfc9731f8c3233bedd496c5b1aaf589cf8874c14d337e24338390c5033c",
    "sha256" + debug_suffix: "a74859121eb431a9947c7455e6cb30b52400ce264d2f715ee01b9e6211e7ce21",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "ccf50190e677cdf0f087e26f4a8457345f8a86713c8008f84229464eb754e212",
    "sha256" + debug_suffix: "e14a01dee741c06c421a70afc1324399623e8bbd10d16af19ab0083a65c14df0",
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
    "sha256": "1589a6ba8dcf089e8d9b89b7c43e2e56926ff8ae35a3fef71d21ed69141eacde",
    "sha256" + debug_suffix: "a7cfb3f0bdf01be250c3cb991536ea44a437feb4c2abeb7f4c4fb55d2f47f4c0",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "9947ea7a42b50f4f4b8f1ef8cd41185fd15218f4800c90c201055cb0e3c4234f",
    "sha256" + debug_suffix: "a5c46873858393b34383c9e1066c359ac6084504ada69dfca3eee9b043c8aafb",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "431a5d53c67891093daf5c7d5b33d7330bee952f3d356ec06084822360c96cf3",
    "sha256" + debug_suffix: "ebb3a9ecc26b772917164e9b2d76abb3812dc78e7e9d1e9c83f50e9b97299944",
  ],
  "kernels_optimized": [
    "sha256": "d956cb7909dfde240a03afc27258cc27a1f693608b137d1112f5e04c192b841c",
    "sha256" + debug_suffix: "f12e091862802a1925218d9f97a9cffff0e167505bc5751457491cdf25d86a03",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "8ff17fb32ed615483dc5a92c43b7bd7a1fb9165bb1c2316796dc3c58f6180be4",
    "sha256" + debug_suffix: "7f99beb54ea05806f43b0250c4ab3dd2f70319a53a7643d293555aa60c5999a9",
  ],
  "kernels_torchao": [
    "sha256": "3fc3e2174b08ed71bf28850fb89ae286e2979bfcfc6063863b46668f63454126",
    "sha256" + debug_suffix: "441b374f84f51b7516c5ec200c26b4baa7d63741f523bbf49d5c70f25d4cb2b0",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "36cf4610556b6a05f82b09b13ced3f73f8317fc763effd22d200ac6488ba6e26",
    "sha256" + debug_suffix: "116c1e176ff1af29b5762b35565ce70424b0d08fd56bf99fdfef9d207911e927",
  ],
  "threadpool": [
    "sha256": "c9c6001ea5dda7e63997448e70b09a571c0b78e9ac72a3d664f9d45c6cdcb8b2",
    "sha256" + debug_suffix: "d7ebdf510f3a9ae1dda18f52fcb83de9756235818ad1857ed917097cfbbc3396",
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
