// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261101"
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
    "sha256": "cc1c596bb3989cb938de4dc50b33a291b09d7fe90c41a4f7ad89503b84271ce7",
    "sha256" + debug_suffix: "02ca653c5508b518084c1b3025f39758a3af894bad31c4cc922169040a68257b",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "c2953766660d66cc30b23567b7104fa97d55b7b53961d683f4a0e567cdbab56b",
    "sha256" + debug_suffix: "4b56c0db03312c347b3ec6c623a160d17cfb8619c741f3853120235eaa217ee0",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d19cc4bf51cb6dacd10505aff86f1b636499447b8569c61876c4a01f39742319",
    "sha256" + debug_suffix: "a9ce6a096ecbf6d7142062425d9b2e0c3ac65927fc5518f28e776bd5e61926a9",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "677adf444ad7c6310510690ee8899bf4c9894d5c294f295c8349554e27cc5f4a",
    "sha256" + debug_suffix: "262e00064b6235af8990cb729b829b350cdb5291f8e73f3166610207cfa0950f",
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
    "sha256": "5b3cb2adcb12dc20a97b4123c580e826837cf4266414d632ff4a6895165b4193",
    "sha256" + debug_suffix: "a8a99c698c5d930afe63e71eb6201d98941e568c91cf43d0354f9984878e4ff7",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "675f8b9058f696d5c9883616330edd51d12acf014608135d42cc397839450fa1",
    "sha256" + debug_suffix: "93653ba9924a0aaa8f03f9846f21aadcb313ab5f1b33297f22251ba27de67f10",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "1d906cce37701ba75a1caccb7a3272be6d621f201c8fe8f54432877bb2425aba",
    "sha256" + debug_suffix: "45dbe966ce49967b5da7a46a3dd5d3ca8efd68069f2aabd88b683a448a00091f",
  ],
  "kernels_optimized": [
    "sha256": "cc85bf8e73c4f1d62e391c2acaf9addfa27158e3d378fd55d2d579257c31d15b",
    "sha256" + debug_suffix: "56114754a97b083885154f1325481541a446780db194de9c3a9f35982cf87827",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "6c0cd1b8442785d2be59b6d8fad04c75d6434a01ef4beed1a7fd79820262eaf5",
    "sha256" + debug_suffix: "3eb78037f84df1ad4386f421071174d90b0362cc8fe577a0099b04f63016d6ce",
  ],
  "kernels_torchao": [
    "sha256": "2e4a2e9fd47ad23d43472449408d31609797aa92c85d230cfcadd5333752ae02",
    "sha256" + debug_suffix: "17d34d02b88b03868679e4bae1239b2adeb1d27484f7e5c69e7a4002122f2e9c",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "42753fe25eb1357c1859f321f44b5ca5d496abcf4e260708c562fa7c38ae8d5e",
    "sha256" + debug_suffix: "2f91452a3511016cbc03b9a2e691c515444bb27c933985b9405ddf7e8434b218",
  ],
  "threadpool": [
    "sha256": "77cc0737f8c4c1c8ed8206f32fb8b7d79fe0b14ab04e5c816b8d63971a49e642",
    "sha256" + debug_suffix: "2d31086d6d92b415ede1d6c5eac47cdebb0a6c01ed45e456aca62d86a2ab0063",
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
