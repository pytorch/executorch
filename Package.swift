// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261009"
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
    "sha256": "e227d3fdf86a83c6e93a94321fc82d6dd6d922e8aa0b6b4e3125c1f1a56d0845",
    "sha256" + debug_suffix: "d01e19a26d95e19b4da3ca54208f82a0f8abc382546ffd1d3eb17c5ea018cfdc",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "02089c84151ac72ba1d90994c6c23bf69f1d0cc19bffe848020d702b019d84d3",
    "sha256" + debug_suffix: "f8944eeadf94747364a501f314f6577f8c73432e43fe5f57f449ddf830245f8f",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "796f5f36de3410998a417cff9756bd42dee6e96293a42932eaf2918f0dd2d425",
    "sha256" + debug_suffix: "01f2eb30c5e18a8603cc5d0d44710c04fb5a01725f92613a8cfdc5366ec0f63d",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "6f4618c84e865f98246a1c1fb8a9886962691b5439e384255f797dd538e1a2d0",
    "sha256" + debug_suffix: "1d3286433834c4013aae6daaeee2de5e9f34d6ac92b3472e354d9fe740977251",
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
    "sha256": "4f9c9ce402b6600263c2cbb54acc8ecb3c58792b504c2d776599f7e189863d9a",
    "sha256" + debug_suffix: "27ea72c86e33060064e47669e987f54f4b9cec8b6c45c33c28bd754b8bb03fdf",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "5d3d4e0865ce85197ddda79d93af76d02513099efbb79b64e9b81303454b50fa",
    "sha256" + debug_suffix: "8144f60d624ac34af4cfce233e77aec7d02dcd4703cbb84700737f5925dc7c3f",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "df04859c5931e5cd9b9a5b0d3b298454269c7dd9be4ee74bb7504c07a7673d38",
    "sha256" + debug_suffix: "105988ac9720cd64e9f8dbf952d511c3b750a38659e339937bfb6d084fe16284",
  ],
  "kernels_optimized": [
    "sha256": "fc46c5eec0c6fb680fe687a2fa3145e4617c97047d137ff4e592ef4cc25e5594",
    "sha256" + debug_suffix: "438aefcbe284108074d7fade623e9e438173e3e17ddf6c5f31bc601f0243de9b",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "2754ef422a74f2a453668fa9bcf312cd781f76ff45498d2d31a1086c2c07ce9b",
    "sha256" + debug_suffix: "418fb7cda86b35f583a9f8e5f293d8deefb6960b71859586df898f3307eb370a",
  ],
  "kernels_torchao": [
    "sha256": "d05cb85dd9e4b454cb4882c64d4e426d3c29982e177c23c3c4d82c6dc8c76c71",
    "sha256" + debug_suffix: "91dd52ef0749d521a52c4a6e9b11587a9f8c750019214ad835b588ab5a4a89f8",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "ba6017faaab3188d28c95179b6bca679ea888cce98591008234b3b084dbd47fe",
    "sha256" + debug_suffix: "66168f8144d6c44bb68a8991ff16889ed28c059af0344663eaa148f3d0e95e2a",
  ],
  "threadpool": [
    "sha256": "b8d0e8d337a790a20719a016e8bb427e696bab743bc2b5a06b6178576f92c6ba",
    "sha256" + debug_suffix: "d4172b969adf1f2a2476a5f2a2d3f1bf5cffe96c755f1304df54ba89e8a7f418",
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
