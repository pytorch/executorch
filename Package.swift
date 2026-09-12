// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20260912"
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
    "sha256": "825f5818f71799b9e580f6e2b3e0a56086d4f1ced755bf64c65497db307afc46",
    "sha256" + debug_suffix: "3b945b588e99257869dd1f74f845ab9c759df826becea05bc9becede4824fb02",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "e48a4ab67aac3ac5048b197d8eb2b4c2edf3e5b808cf1f7f0b0e75fcc394eb51",
    "sha256" + debug_suffix: "652c6179fb1845760d4039c48d37a02ca6e8cf53894d4c278b8568d6e8a476c7",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "13b7f0a4de0b93d157da0f9f2bebeae23f92913d9c883639d868a48840dcd3d0",
    "sha256" + debug_suffix: "70d43afa2a2a1719eedfbc585543657166a78ebac92c75b08e3c4c21b9215bde",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "028d04fb17d4fe20c068d1789c113581db01f9f38ac8b64b75e62bd0494efb82",
    "sha256" + debug_suffix: "6f82ade953f833bf7a5a63dc2788d08db5f052cf9abf576de598cede15841d26",
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
    "sha256": "26fab388b513d669f5e2d3974e147e5d3fd48e4c9892ab4bbffd4017953a1852",
    "sha256" + debug_suffix: "46f9d6c59fc2a335aa583fd420860f588ea4116f8b0f90336f402011f7a3d3e1",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "fc9e00e3ced092c64e9fbcd1fdceec50a34ed3e2331d7f14d763e1b06f4e944a",
    "sha256" + debug_suffix: "71291013638ec24564fb94ccb8819dbe14ad36434927f1e7f623d7306933eeae",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "7ebdd2dba10ea0235e5935da3c7ce9629c428687581f031245c4d54f415487d4",
    "sha256" + debug_suffix: "dd993d2779a659397e4220f91130fad596c1a8c2053147372971debdfe7734ed",
  ],
  "kernels_optimized": [
    "sha256": "4d12ba2656a88a17b1cb750140c30fc6a201d0195e918d4c28cf70da18f2e0d4",
    "sha256" + debug_suffix: "a6325f029c2f1149df4872b2c5f8ba39e1bcc9afb6333c2921ce8dd4cc86e3b7",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "e61d0fd02e2b4b95bc468bc5be2bfa9c90be77cc9b7b89d1a727e004e0123f9f",
    "sha256" + debug_suffix: "f822199cca922692d244cc18f2a2c9b431a7e58818225e9432c62a74b73bd3d3",
  ],
  "kernels_torchao": [
    "sha256": "f44ad36cd772f4b9b72ed9ef41f52f9088367786fc1735123af8d1d8781e783b",
    "sha256" + debug_suffix: "79117bb6c32f4a23fed9c6527311e85dfbcc7883f280da1d90553c7fd538cca7",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "89c33ab2be15254d1d70d98fd4f06fc8f09a6c05d95fc0ee2b666a72a57963a3",
    "sha256" + debug_suffix: "0a839993966585e61973137e6b2aa1e7f5091726bfa254047d185bdf53da5d04",
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
