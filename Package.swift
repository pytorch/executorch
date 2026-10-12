// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261012"
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
    "sha256": "5a980b2f47dcd6530e4c7d801b77ee23b31671cc898342afc050cb0a969d44b5",
    "sha256" + debug_suffix: "dfad39af0906138cc936d47a5e0b2248fc357490af1b92b54a88e362a56829f7",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "6847d141b2024b941364fdf92c7a6607c871a38f668f621d3b1c6555df90de38",
    "sha256" + debug_suffix: "800c67067fd35a9e435b9cfdae58fa2720e40084f74f047012bb24cc5cf17867",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "d20ec64e8e906de218fdee2fe48e8de76c50b7ba009672aca41fc7cb3c56070a",
    "sha256" + debug_suffix: "ecbfaaec025b2cb59c144398abf9140e592c1565a667d27c4a42113442ed7975",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "00e2aee17217a2001c6519dc95ad6f07d51f5bf0381157f22b42aa3d0cc3ade0",
    "sha256" + debug_suffix: "35daf00991d9d620d68a8bd65e8881d9cbef2aebe895e37a455ea68337cfb485",
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
    "sha256": "b70c60b5f3a41847993588c613979656b504149f5b1befc47e388d27af8563e0",
    "sha256" + debug_suffix: "59401c4662bd26c855ebed885610c47943c9b5b84705d532e22a9b637703de59",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "c09ba76b0facc1070a6455ba47b3659cfb50437510403c1a651eaefecd807485",
    "sha256" + debug_suffix: "3d6a8ce8d306a07ef9b20aa3b0ac5ec687cac4d2ae8060984680b383a3c425de",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "6fdaa748b7bf6c6f6b1e3f103cf02d4e713073b98cef5aee0b5cf3fe981f752c",
    "sha256" + debug_suffix: "a0aa4d6c597c4ee92c6171a7455b25384b2d84534c0e04c0e5c06abcfc464883",
  ],
  "kernels_optimized": [
    "sha256": "a68df6b0d7abde44f16f907c26e01e3da2d531c557a649688b9c0bd39750606d",
    "sha256" + debug_suffix: "71654277db47c9572f139469a00f74f2ca641ab1f8bf0f5d984f044d3f71ad71",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "53390a8e81c8fe4439a46f134ecb2726286df5a8b2b471ae2edb63efdd6b4dc2",
    "sha256" + debug_suffix: "9eb4e4ea613a828c79691bd32a7b1613b85c63349055cf904a759350d3ec351e",
  ],
  "kernels_torchao": [
    "sha256": "bf8f1f3bab92405c06dfa523264ab650995f10f0d69b7779b927d78caa3cd6f6",
    "sha256" + debug_suffix: "184c97e89a03998642b10c3e14cb2f359ae015d73cc5e60362139091074955cb",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "1d367366b56748efc4c0ffb3b0309572613ca6a5a4a9dbaec0b81d76b25cb9ae",
    "sha256" + debug_suffix: "a4549f4a42bccbd36062443706cb2a76b633b2a60645b71e0dffc32ca41cca3f",
  ],
  "threadpool": [
    "sha256": "fe998cec8e56a022470532dbebc006236efcb3eb2d69cbb9fe915c2ae0ec5f55",
    "sha256" + debug_suffix: "f8a7dd95241707bfb3e14c4507a36b9818154f9aa2e03542607f13df79e097e9",
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
