// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261029"
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
    "sha256": "80996601f35b74a898136855b4f12980bab05b95574457037d45f483391203d0",
    "sha256" + debug_suffix: "5c85d682d83cbab1d73cd365e7e74818db2726226124e3fc40dd802fecefc7e2",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "08460f9808b60812beb6fc332073cb905cefdf081ef7f1bb582e8299e4cd2add",
    "sha256" + debug_suffix: "dcc6c7479d0cfdd9b7a65297a26dfec1deb0c25ed8f53ecd53a3ce492ff22089",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "68dbab36c3e3d0f8b5f0f69dd783ebb5c460349dc15aac700efd65c5ceb12dd5",
    "sha256" + debug_suffix: "32a1dc6db36ca14484da658123aac8335e4850169c74473518c4de10924bafba",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "a3511bbe0300fc8f61c67cf392ddb8e93c17c9b33d85f34a8884e00a7c16ec00",
    "sha256" + debug_suffix: "80e6ab3389b5bcfab8decb464181a51ca57be87c4f5ddc106b024f73c97bbc7e",
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
    "sha256": "eda02329a40d0d1721c41d56aabe6eb888691f3fc29aafdaa65e6e9319aa50c0",
    "sha256" + debug_suffix: "266181a25bb6f1016464cf6d3e80ca72a8a46d9f1b9c3b69159eb5fdc25b0d59",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "48fdd44a40958703d1642b8834268b4b96c9dfb31a0b2062711470219fb0bb30",
    "sha256" + debug_suffix: "136e7203ed92ebc3d30ec73925aade79ed57abbc2fca44f990e4de0c799d6ab3",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "0dd8b9b6a0271e8226b517e068df9398c0ad567a180a2013728074e6c87cf497",
    "sha256" + debug_suffix: "3d69eab85693bc2d57c64e9f925704f56a2cb6ee80b7b392fa62f7ab5170307d",
  ],
  "kernels_optimized": [
    "sha256": "0e4fa861cfc0f6a442272986625a96c8da73d5765250f0f01aad20387341b0e0",
    "sha256" + debug_suffix: "f8dac98a5012ccbdfca0d56e587fe6c7ddd7e9b3433a07b5d508ee2a0022c916",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "6aa53aa8c9f64d187eae68e635b7728d89edc2c70893a6279f6c5dd9d1ba1c03",
    "sha256" + debug_suffix: "c33ec9a0c8c12b7f03a24978390d912d55c921f0e544d3321dfa9cd4c7b6f63f",
  ],
  "kernels_torchao": [
    "sha256": "4648d259d5b72f2046b6d2119214ae69d1cecdc89ccc0f5220ce0496826d4a47",
    "sha256" + debug_suffix: "4075f434a3c07d3e1e9a020dbe840a2246dabb184be37fbbabe84d869ffe24c0",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "281606eb858831348e29dc3f31694aac387ef2a9da894285111af5a70975cf52",
    "sha256" + debug_suffix: "bc07ea8b5be2625db37534ebb3a36efae95c0fb61358866606b9051fdad48a2a",
  ],
  "threadpool": [
    "sha256": "df85ba9a5dc482923dbad634e3d01fa4973f88f282316f921d2a52b96c264a1b",
    "sha256" + debug_suffix: "8f8bbfa2332f9605ee1a3e5f2370f44837cc7f9753124590a715dd1d040a6d28",
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
