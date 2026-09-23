// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20260923"
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
    "sha256": "d87a2c487ade069cf1290b77168b4bd7b803cd1fa204616485d078904aaed283",
    "sha256" + debug_suffix: "4321f7dc19ce2f35958bae0390eb6b3be2d3e9d48c842d4c4e39954a5181c4d2",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "32c7755300c65c8ea86f4c04865e4826c5df2b072b47c911d874f01a53f82000",
    "sha256" + debug_suffix: "816e1aff055bf2471e250d920f2a122a857a9df2c20f9aee95244af569ce4faa",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "dc0cdcf2c21ba6a34c91dc4344eedfadd31995781b599acec859ca7fb1f5e395",
    "sha256" + debug_suffix: "7cb3b6fb4501a87f56b16d4b7cd735d673ed310e492ffb1c72715f259749caf9",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "fe0355e9d29f92586e367663bed20fdf4f82bf9bfd79390b8cf6e91d80afb46d",
    "sha256" + debug_suffix: "2c662603ceef876b9cd03a9673732489e2ff9029bb2478156d6741f7e6783f76",
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
    "sha256": "c9a67c3cf9409d043a459fad76b3fedd0669bd7e64078d57438c9435a4df8d11",
    "sha256" + debug_suffix: "17027982c219e580adca26a5c1e2dc2f13a940524a5325ed672af43904a516aa",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "a50ecf53b62c8a5bd9e3bb204229138e8d54f4b0c8c973d37e21a3ff4cf9983d",
    "sha256" + debug_suffix: "b43818a787e8c595d3df4e14d0c23d3ef427f0e3a5c32c3fb82b75108eca7ba9",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "bf6219054985939bd7a0d633a000d9d90c1ae6ea075b612214b82ded7dce569f",
    "sha256" + debug_suffix: "7e84d07e0889b297082ce1f174a72bf6b36f719a9763b4adf099f5f7603eaa38",
  ],
  "kernels_optimized": [
    "sha256": "660d4c8b50e0f156410887f5d2afbc69f39cfb5e6beeb5cc2e2f5f1011d682b0",
    "sha256" + debug_suffix: "44a959d306048c6b272dbdc49ab434eff706efed53cdd3fb0c4bce9ca99f17bd",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "5e55945a343bffa3d1e163abc4c115ea1f0231acbb30137dbb862438713fe21e",
    "sha256" + debug_suffix: "ea3a3d1dec4325ff9f121c34a9b5c5c999a004e4eedf847bca881cfe5416226d",
  ],
  "kernels_torchao": [
    "sha256": "ffd97a8e85558b771a7e6262a27e4e1227336edb03849a2fd4af89b3c20da9e1",
    "sha256" + debug_suffix: "c5e85cc1abadea1e224ad76b8a9839cfb85f0c0b215434e3de235b2b912e47f3",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "ba2a623f55c775e1d738549b8a2435f663bc156c6fb5fce6835e5c841942b360",
    "sha256" + debug_suffix: "df1018e9cdd589ba60d37e5cd00f6e11bdffd537ec23fbce5fc68a7f57693386",
  ],
  "threadpool": [
    "sha256": "7cc0fe90a554b243f95a967c8ed9af9012960bb4b19289fab3da10a42bb3bf86",
    "sha256" + debug_suffix: "ff63156b33add9769215cfb8201f50ee29ed5a176feaa3ed44538d7f76984765",
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
