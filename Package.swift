// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261127"
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
    "sha256": "fdbb238c1a13f472aa72652fb22b9bfff0714623ee183b1ded1ecf9eddd01621",
    "sha256" + debug_suffix: "43c4ba275587793334cf16730411541fae442edd7500cd8296a0a34ef81efe29",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "b4ed1f975a94e1d450808aadd1e0fdd0f6ffe348049f55acb45172fc4fff23a7",
    "sha256" + debug_suffix: "9177b11f5145e0b0a699ec63f3d780db4c26cbac2df9ea7f8d2b53406ba8ccd7",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "2c2975b8ab0c86a101cd2a0fe6c5f08b8e10f97d8e5a110fc219312db8e88691",
    "sha256" + debug_suffix: "ecd843f194fd82c4014c1e7b9748f8250324faa07563bb3df642b8508b22d7d6",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "508b37bf0cebbec817fc8933941ce40537d7a49eaa47549fd2ab98309a742da3",
    "sha256" + debug_suffix: "5896ed732e4870112ebec0dfcf2739fddc4a48e4d60d94ca148ae53622614696",
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
    "sha256": "6f26960592df389751dcff84b90c253331556536d8fd98e32c46ed7fcf6cf17e",
    "sha256" + debug_suffix: "a378d87ecf029d51c4b496e61a4ef7e07e9aefc8c98989ebd83408cd4ce11e52",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "24b6f514f7598921322ce42988512cf6dfaddd58ce2261f744fa6a2ef3183000",
    "sha256" + debug_suffix: "f5a4f9360534126b9ceca04765a8ffe8aa42e039d0ffcebf7dd32f9403cd1fbb",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c31dc456d3c1badffac2b60e0ee3b9b3b24437496e9eb5ea9c5e196475178fa0",
    "sha256" + debug_suffix: "c6b519f65ae6834573fa41544c4c850f2f6b88f974abd8f78113473e6509241e",
  ],
  "kernels_optimized": [
    "sha256": "1565efbbfb777f47303dc4a660324cb49131cb0fe0011a7f0b044675c3df3b43",
    "sha256" + debug_suffix: "68b573de9a012640490d927d9f6df09863cbf633ff913d37f8caab8d8765f982",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "de251af79007f81c4a44dcdad5b19e9468f3a3e0f475ab806616f15482a48a19",
    "sha256" + debug_suffix: "ad0e43cea827f1455b8584d597029e9080792917dbc2bc33d578ca597aa6a899",
  ],
  "kernels_torchao": [
    "sha256": "d84787fbc7353a98bff635042e3eb7e2eb6504f4946b7bd5dd216679ebbfc689",
    "sha256" + debug_suffix: "955ed1ad0165c3438a5ccb442aa3fd6326a54ea064961477cf154d0f679820b6",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "15837fb70646925c4b7e20e52bb1f292423a5acbd6d09138194aa65db083e461",
    "sha256" + debug_suffix: "ef695ff3cd0a4320f292983f10c1922502cbc783187e6793c556bf87243e9856",
  ],
  "threadpool": [
    "sha256": "7580f4c8e37f077f2eedee47f6e06eecafdaceaf140272fa1009f334ea66a027",
    "sha256" + debug_suffix: "1393d41df82d1788f251d3915a507cff1de95739e99c092bf6303ed70d0361c8",
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
