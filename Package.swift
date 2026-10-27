// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261027"
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
    "sha256": "5dd87c02bc8e2a79662c6f98fb2e24f7690a1d979452a76e55dc95a133132530",
    "sha256" + debug_suffix: "00c822134d688861b7fcff12125a104606393c5ff57492c2b32b0627572f013c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "7eceb26417ae4284eb1f7533dadbf108d8b701987258b40bcda416882f17d577",
    "sha256" + debug_suffix: "5651da56da1149b0a40bd911b366f735b5964f1171a959f72b4683646f2439e1",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "47364660d67392688edc0252154552fcda5ab111d581dd24902b22ac2ccc35fb",
    "sha256" + debug_suffix: "b449d14faed4ee95897e8e2931fcb962acb4f301d2c832cd0d393e964b61d882",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "09d4d6aac8bc95eedf19ab52957d6312f102a6bbb0fbd3e61d794a14b2281032",
    "sha256" + debug_suffix: "1acc89961957f943c70406c9013ee3fe676f2699c6350df65375c54f8077b485",
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
    "sha256": "cba4391104a81ddb6632f98613d3cf5b452bb897870aa4e9f2c8253de1c6d655",
    "sha256" + debug_suffix: "ae532d65a6a3e3d07ba6986e435704f94720cbc39838ce0f910453455f6a73ae",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "b55288751de316074483063f90183d3f2078d0ac7a814086365b3871ca2ff65f",
    "sha256" + debug_suffix: "7b39c5cc06e1c471213b8c7e2fc2ff6d0fd6ecd2e7090e5366a72d09f3d19a0e",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "6c364f6881f17142ecfb295f395eb510f90b72422a049a475391114b7fd4e45c",
    "sha256" + debug_suffix: "41b129265f73c226c9ee8799f8d37baea3061a2d808cdd02ac0b0b76f41ad78d",
  ],
  "kernels_optimized": [
    "sha256": "fae9585f3681aaa92bcef140603e7db084591efff32bca74cfb59c4576d4b696",
    "sha256" + debug_suffix: "10559bd5794d4f43016210ce082b45c2717826a435e78b9f811801c2d8dba703",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "cb7faed10bdc925213d4cccc7604c4b55b603c7ba2bb3c012752249e82af2043",
    "sha256" + debug_suffix: "6a38c21154c4f61f146c3423d128f2c32513f98073c54c6d71faa7028e1cf96f",
  ],
  "kernels_torchao": [
    "sha256": "f5848e581a0ca88a443705ab9b1048ef08af38115404317f59c80580859cf1d4",
    "sha256" + debug_suffix: "37a4f99f4dd8ae2ae8827c089ff4c09db17a819388a7d5c34ade7b12b6c1b36c",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "1299965aa210fe10e98f6f49a984d2d50f1ec58afbae9c30ac8eb1bc4399089c",
    "sha256" + debug_suffix: "a668516d4ee04f5de5adb6ea83b5ef6ed34a13a2fd1fe6f2a0d5bf90c419e9a2",
  ],
  "threadpool": [
    "sha256": "270a1ebda4b6a41f49219d8834d048d2bee612e2d3688752abc50e6122bea285",
    "sha256" + debug_suffix: "6fe90d4254a3a72326515ddf21f567d91be7c9eb70894782a72209609fff0d27",
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
