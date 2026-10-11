// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261011"
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
    "sha256": "070c35e4a9becb748683100a91918065e49c6f30de23b3d0445e339621dcbbfb",
    "sha256" + debug_suffix: "bd07955f3c7b3c1219a32a3ff3466e3532fa60c788a12b0ebca31e3b1636dc03",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "02ebbeb89a4e0a384113db526bab2fc4494d6a5df862a508002e2fe3e532645d",
    "sha256" + debug_suffix: "49894029386af271ca92cc4a2d941c66d2d133fef3e5a1651a98d4a3b6fff2f9",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "99dacc2fe1ba6c7acf7220609cd8f6ddfd4912258f6ab7a8bb4c507a83f9e2f0",
    "sha256" + debug_suffix: "35fb3bf3951287b2ee769f81eba4558af63fc7161fc69f744760bcfe06879b3a",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "19d088912b557974e0c8d55c70681565afe3e7927cee4dd981ca514a70aeb842",
    "sha256" + debug_suffix: "ad23742fad89a8fe06e459dcde88ee76b03ca4433c5d42c8d99ff05d358e2683",
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
    "sha256": "1dd6df9a4ae22fc351d7511494724611f01562af29726dbc3171af30dc6ef117",
    "sha256" + debug_suffix: "2885ae5fd974449c2a6fd06ed71d9edfe3b2945acf873111da437f6cc91e9bdb",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "7d56e78220554101d2a321ff4c5dce2ebbb4e5ad9cb97d7451c597e10f054737",
    "sha256" + debug_suffix: "983ebbcab89312deb7e15f47c3de121315c75528e121c11aadbcc5e29c307200",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "6cf2153366cd69f6beac01a9df0a40c1375c5db9ed35ca0e8199c04776bc8073",
    "sha256" + debug_suffix: "69fd423d0362ace2d90b7633475632d7208581d821ac436d0f94e6b5428f8404",
  ],
  "kernels_optimized": [
    "sha256": "fd09c684170288d1cadd16c8af3b9b55b459da896c0c4c945d801093761cab6d",
    "sha256" + debug_suffix: "08af5f3c4b19f83cff2b0fe45f6e0b74520fe14c134f5f907028bacf3ebb2eab",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "3cf1934c998ed3a73189afc03e2c2efc4907a58f955308f437d8cea857e85754",
    "sha256" + debug_suffix: "336e97bded926b889f894072a331e8e41d9b38faabc489411fd636df5084a0a2",
  ],
  "kernels_torchao": [
    "sha256": "21bb5562d5913ee4bfbb1b8fbcb0915338112176731a104e917d3b1ce2de4db0",
    "sha256" + debug_suffix: "63e87616ad23987ace3314071313e73afb4aa56fd6ca014c80fa26564e2de8f0",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "43dec1634b730799856231c6ee0c299b5154883d6dc2a06e652cb0adaac4ae20",
    "sha256" + debug_suffix: "be300a74984c1b7c95651d49424ad1ca9ad234747a9c7871f365ded166fc1229",
  ],
  "threadpool": [
    "sha256": "54c1d695b314c818109ea7af0283e7fa63442fae31265b4095974f6242de4f00",
    "sha256" + debug_suffix: "daf1e8200aac99d68ca8972cdcef79a1223829d602cdb6a94b769d9e9890e5c2",
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
