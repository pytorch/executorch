// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261015"
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
    "sha256": "404aa669419bd50c4f8cd91444c85a20f12a66579959deadfb691ad748493c58",
    "sha256" + debug_suffix: "8937148e856e1e6dcd04df6c612f7e05f0106066ae5144dd181622d1e2541a11",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "eb5ba63a54ad19a75ff3b4636bb6a1f82fd5faafa4ab61dd84602d4510de1eea",
    "sha256" + debug_suffix: "1ece641d2823b9d6a90ceb45d6c5f3fc11dbb100789ec30653264da40969d56c",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "746adce6ecc2945c432684239016b86b35a741f01ffcc1cf752d76533d19b8d4",
    "sha256" + debug_suffix: "9a8cba567944a95390d895af80019ecbe896884eb087c530876c1a78a59ae628",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f8c807629fd260a32e41914e9f3f0e127f0931af7c8d7d3c25f3f3dd21c7b54b",
    "sha256" + debug_suffix: "a989fa7de6c8a7348aad4a3dc64be7d4d94112dbe4140bfc5a946e73f881a32d",
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
    "sha256": "d38f87ad217a8a5b10930dcc3b039932205d0a16d9ec3adfc9a699e6e27ec130",
    "sha256" + debug_suffix: "df332fd57f385345d854c4b32b1ee32995e3359e4e922fc5b14557ffb01c8ba6",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "1c9d7a2bf816b84650ffce2f7120ccff161fa5421e72541a189abe90f2566047",
    "sha256" + debug_suffix: "8bff96981cfefcd93c00fb22a3c9c64cc5410b367ff2ced1232514c6ea2cef68",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c41a965ac61a22df1eabe9f851d923129554a88039b06c085234ae7353635ef0",
    "sha256" + debug_suffix: "94a84e69198ac9bf85d9c0ea3f7fd6ee98ce014714afe0f4e6f09eb72622daa7",
  ],
  "kernels_optimized": [
    "sha256": "baa3558bcaaeacc42224b998093ad8475867f8a65fd91957d12f5fd888546d31",
    "sha256" + debug_suffix: "251239f23023e92b07e317ea6c555e721083c22b54daa7ec6576f54093c44e3d",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "f1f4d732839c649cb96b5ace178802a7410e83ce4f796d86b9b53ff21cb72dfb",
    "sha256" + debug_suffix: "8a37d71f553f9971ee6b6ffdb2830cf2c640c26713b02001504540a5f979a95d",
  ],
  "kernels_torchao": [
    "sha256": "856c2753011ba1102d2fe446efd78fc0a886f949ffe8376d24694faafed3a433",
    "sha256" + debug_suffix: "e60f08ffe3aff6852938f594bdad44544e5c9bb4878a3be2668a06a83553b589",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "f5e46df67ce6d72ead350296550e67a53bdd85e7e5a6da828d6a97d916a98944",
    "sha256" + debug_suffix: "7c5efc4c57893ed77c4518a2acb59378e6907effac01e5e73a055d08c5285d4b",
  ],
  "threadpool": [
    "sha256": "91675729ea5331c23c62bc84b39ccea80fd896b22dfb08a8277d90b84d3f57d7",
    "sha256" + debug_suffix: "162bab2612a08d5e11a1c9d203e6a91ec2c1ccd5edbcf28649fabbc18b89157d",
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
