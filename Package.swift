// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261019"
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
    "sha256": "6889ef2c162fabcef66381288ea6092e0e44c2346ac2bb1b3fa84c3d0111e41d",
    "sha256" + debug_suffix: "d6ed4dafe3a13ffb5882cdc0c4074c7d09a77e0526a8ecd01901714598d92480",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "2e6fe30aa1af00558697ca7e5df1f2eedac065cb72bf957261cf1c8af02dc79f",
    "sha256" + debug_suffix: "987ee0972102f44e7f13d97eae98c7dbb3cd21eb9d606a2555a8fff2d5212c03",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "9d092c3a72f6d1a1750d68808d67a9479502699233a5129c2ac0d64cfbea0bea",
    "sha256" + debug_suffix: "eebb1476e0b2675c32fb7705326c90280d5aedf9da1ec88ca461e842d32b358b",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "ffa8ca87f99eb760c8c92c1a9327c237180604c4da8c4c9d691913889de3f441",
    "sha256" + debug_suffix: "b6d339b502709b58edf8d45794c024d3801b1f5786b796e45736612d14653d89",
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
    "sha256": "4230a92b5b87ab161dee6fc647bb74dfdd6efa684bf2c862f776644dc9e17094",
    "sha256" + debug_suffix: "150ba75285ea9edb9c5b91bc25da5daf4ddc127bfc14c6f7c99db38b48570309",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "c579c3099e86e16bef1ec7928d42c1e549755c3535115a435483527e9ecb7914",
    "sha256" + debug_suffix: "8ca817c7d05647cc7bf3c45845f5f13d951a1372ad95410599c402f345b5915b",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "9f83f586c44e0838fe328fcc4d0ccc86d0dbf600cc2a31bc46dc5b3d80874f90",
    "sha256" + debug_suffix: "671727b5ff4126e2ac1e208dd1bbcea35f51e0a4b1a5c03e18fac2bf9f80c02a",
  ],
  "kernels_optimized": [
    "sha256": "17659b3db6f356b9cc102a4037144cb66775f32b88199c79ee659cf1c727a1e6",
    "sha256" + debug_suffix: "b0692934c8246c359f8e9e1b23f20abc75c36a1b025f7efd6492a42abd5ae527",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "88a031b0f3597207165b8981c9cead6d4c7698c7d7deabf0e0bcbea1042609fc",
    "sha256" + debug_suffix: "7865cdb63cc2144e73ab7d3fd6943fe44388b4a434e8e99fb624f3b43fe81d24",
  ],
  "kernels_torchao": [
    "sha256": "18c692bac94a3ee78173006827644cce25461dc23700be02dc7ec39edc3c5dda",
    "sha256" + debug_suffix: "24c2915b28b1100cd02f7a4757b965ac094c6150ae657c39fba06588ed3b61a0",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "ee80412dc081c961f0a57e2594b895a9674767e7b32cf57c12b21e0f0202676e",
    "sha256" + debug_suffix: "88825b7439806889b8b20f2deb19d4bc3cf0a40cf626bf6b13a0b5d360331aa7",
  ],
  "threadpool": [
    "sha256": "0bd06dd17f202578ad26a2befd45c27a6a4d6a67444dc88e1132c216737f3f47",
    "sha256" + debug_suffix: "ade4366420e3dc848fa46d1aa1a285af07e70895e7b49d2c330894ec52a533f2",
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
