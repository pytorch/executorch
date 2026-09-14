// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20260914"
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
    "sha256": "c4a96bad92caff3099327d5aa76e744bb71fb0a2d062b308817896071603c6b7",
    "sha256" + debug_suffix: "0c356159db543111b014cdbc92589aa3f2d07a4dce8e727bb25a66c409d1de92",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "24e746e450d27e6e17fc7cdd3d2b505b9632a9d94688a3a29fdc1b54982c1498",
    "sha256" + debug_suffix: "ec357059b88eff4517340647f7c674860bd91b5f4be15fa714ef563515a0c80a",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "70bd8b23efcc1c924614b1fb329226e45c1e509e02e036e7b7c011455dcb5993",
    "sha256" + debug_suffix: "7bdb9e1a145a192fc8475c6a6235536b1301e78ef7751d183ee1ed2ba14de779",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "eca6a2e600c9cd136603250c1959f95972e1f1485162e8336ca46a10159ab6dd",
    "sha256" + debug_suffix: "763f6bc84f01e8a735a9fb9599e8f6ee77eaf7b4b80119692b11264da3f4a8a7",
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
    "sha256": "4bd77595862530eb885ac3acd14d82b94886e0dbf30cf49ccb065ba029cb8ebe",
    "sha256" + debug_suffix: "e2c99bf77d140d9651de38bcf70efffa12d51ff1d8563e3d5683f6dd8ebccf93",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "56834de471488bb69964ab4325fe897282ca6f8011da6166fc062bcaea30e5f5",
    "sha256" + debug_suffix: "ca3a93f8389ef7a64fd2bec4e2cc2f3fad817f96015ea407d78ff3d9d8d88128",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "866665cc5b2fe698f00fed4fe9fc54bc605ee603d63666e84dce9752b1e44444",
    "sha256" + debug_suffix: "8f313aae08f61d2839c6d463ea6d4127ac1bc2bd695f396574d38f9b6e38c427",
  ],
  "kernels_optimized": [
    "sha256": "b8679268f2fd188de384130f6dd882e68580709fb53cc7b16f0a143cd07ea534",
    "sha256" + debug_suffix: "7544b8c844bd8aff8b42ba8b927544b729142032885277bd1efc01b3a2ebec6c",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "cbc11bd861f1cf10a10bf23b2d48581309c7695b0683eb0239fd9bbb414b3e60",
    "sha256" + debug_suffix: "018af467a82d0f8be5c3d4d929a1f1ddec281cd5f237ec15bc3447c8e63cfb7f",
  ],
  "kernels_torchao": [
    "sha256": "0bad3773f604aac22c8625911d5d0f5953362515eee0fa7f3b894bb61bebd124",
    "sha256" + debug_suffix: "73183094b7e99937f3323507530992806c0b47723b96a1dd52916e4d728609d1",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "cd3106c64d5779744c75a1f84726c97fc168816cf812fe60b31f078432f712bf",
    "sha256" + debug_suffix: "fcb5b3baa11315e8c93efad6fd878787dd44057e767618f4bda0361fee2f5250",
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
