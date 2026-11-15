// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261115"
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
    "sha256": "6980c5070e72f9f3afb015ff577b368bf2ac5a9cdea533feb2cd71d124b2ec0c",
    "sha256" + debug_suffix: "70733a26860e6bfc012c848de65dd250a059567f47c0c577357b3563f16209f9",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "7ba3ecafa1e8d470242d060c1f68dce56311f65db64a1a551231f084451bd51b",
    "sha256" + debug_suffix: "1fb7bd84b49771eb7b73f65883e42abc488c821d098576db5f3b5740adfb4068",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8a49bef00f68fcab51bfa595915a59c44dd6baaad771d9adde71910ee10303af",
    "sha256" + debug_suffix: "b2b9dcfbf50c9760bb8e38b9b22e400f7694888661caac8d2c01bf1dcaacc108",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "2c6e84b579d3f1ecb52f1bcc674a2221d563213c3c311afff79b125658787527",
    "sha256" + debug_suffix: "d3c38193c7d10174182ff6396ed3584ba63dca5c1f90ce60d17dff1caeeb5d41",
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
    "sha256": "d81056ce2ab8a1d4bc23e405487eead34d621424ef81174f3ac44ae8050533e0",
    "sha256" + debug_suffix: "3525bdd473250547be3af8dd84eacfbac2d1486243fae0239327cfe70a61d4bf",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "50cff08dacbe381a9450951562a1d03c943b855a76b835866c34750f9ba4978c",
    "sha256" + debug_suffix: "d64ed57ac46c7a4f3499c4d4c6d37f4e798190a3d250db425df90474e22279a3",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "2dd5bcb628cae3ddc673f08e997f2bc328587cb1297d92ffbb7b7f1e38bd01fc",
    "sha256" + debug_suffix: "111cdbc285f3b5189bfe09904c89861907e90cdc933fb244f756dcd0ab0a7d31",
  ],
  "kernels_optimized": [
    "sha256": "2435787ca6d3d7093139f3eb522184892860fb48448da7eacdeb943f4ee7f48e",
    "sha256" + debug_suffix: "0d7df1da4f98ea2ded129b56e6267cae6a87c8813b204ea765cef9c893fce079",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "686394c60097c48d1d93372c57df27432d0828ec3fabe28fb86ad1f9f6c014eb",
    "sha256" + debug_suffix: "b2d4aa06a16326e50d1fc7511f61677121c7bd5dc0ebced76bae64087e51e829",
  ],
  "kernels_torchao": [
    "sha256": "cc7bf866d8d7dd26aa698c6cb5057f65f3afbafc1b9027d1aecb5a34016b701b",
    "sha256" + debug_suffix: "cfc25d1b111641137514123890cc3e3d4eb8d73aece5e3db6872525da568fc0e",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "a35b61f1799812453dfbb6c7d1dbf6e019608e30a7a3eaf27bf67e35437fb89f",
    "sha256" + debug_suffix: "2241b757f1579901a0b913f60f50bb9aa03b19d911c884765bfc6739b6fec876",
  ],
  "threadpool": [
    "sha256": "b64fe44c071123781d29100c3206a8230803e667d5c91fb12a8f81b880d04515",
    "sha256" + debug_suffix: "4ae34a63383d666223adb2c184275df5e336d0e030ac16465dfb10d9a034a3a2",
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
