// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261002"
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
    "sha256": "b9265a3f45b2d5a23abb673911d778eb3f550f70f95b71939434272cb83089eb",
    "sha256" + debug_suffix: "8f06cf63600cc53457788e517843ea94c0776fa976d113f60b6e37f4f0722b01",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "d57739f7e5dfe2d5ec0536d232a25071e06473cfca559f92ab75c5351fa67e65",
    "sha256" + debug_suffix: "92c58bba39e56d8fc9ebc044b31774e20ce1d5cc5ba9965e48963c7bddcb28ca",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "773e4e15a88eb36dece1cf5a91c5b1a53430864c55b7fe84b68513bcd6278816",
    "sha256" + debug_suffix: "b71abd1cfd6952105618d635383b8d70b74685610e10b64517b2a3a02a21973c",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "1dc8ea7758b9a8c9cbbc9edb184caa685c9f2b6d58eee5fd62ec56e50f2a2fb7",
    "sha256" + debug_suffix: "4ae005cbff26324ce07a9048d73a9d2a6ca50d19652dc3bf5869f7a264de88c3",
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
    "sha256": "bb40707cb7ca857fcc0f9436a36c5447504345138e4a6fe185101cab3f78baf9",
    "sha256" + debug_suffix: "146e0a3d75a094b41129d33687516eac02d38bbface4572966243246725e1087",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "f4b86d56db954632f87a067d447a183580d4edb15034cedf9cc6b741ca9ab651",
    "sha256" + debug_suffix: "f493eb9ec33ecaa9367aef2d870b52ca1efdaf9b25fca19e788b5ef9687de0b1",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "dd88b76947a105ddd39ee8e22f7eb7cfce952271a85a3166e853df278477fa43",
    "sha256" + debug_suffix: "dc04ab0f3dd9e239e6a951f852415482c6674336158f0b75910eb0046a93e361",
  ],
  "kernels_optimized": [
    "sha256": "2bdd8c19f3bfa5625720fa21f220f14b07c6c841215166cb53fc725c8f0ac8b8",
    "sha256" + debug_suffix: "4872e15370fb6f50b4be07db7d712624bd47252db3cbb87681b1ef3663b939b8",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "cb815d450b804b8bd29721299b5072abfa36fcf86cd047a3e0a7da0731d75384",
    "sha256" + debug_suffix: "ebc56186b18c5d3c53f24fcabe231f1817d7441de236e58dc60d59baecd04e6b",
  ],
  "kernels_torchao": [
    "sha256": "3a4d159ee3ab78e4649ab8605757062f01719c443a5bf7359f4b8d1b07e1ea9a",
    "sha256" + debug_suffix: "3fff74acea17a115e971e24c4eb221034874484ee37d945bbf8858cf1ae1cbac",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "73d4f9b34af59fcbf0f708fc5aa480fab7b5fcfbf22ce3559518f499095134b5",
    "sha256" + debug_suffix: "aa508f9dfa1004fc85f3053d8a6002c47bf87cdb0f330071fe4f60e99d4261db",
  ],
  "threadpool": [
    "sha256": "2f2b434d93aba368f56c2513c507f610a8ea35d52c59474bf500da6935965f1b",
    "sha256" + debug_suffix: "92521a4e7028a47ce8bc917b71e2c9c24713ce7b9c4e8309d3529c517e88c5a6",
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
