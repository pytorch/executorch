// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261026"
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
    "sha256": "d52f5529b7c1c2b69571972fe94bbaf5ba636d99c34ebfb5b2a3aa04025b278e",
    "sha256" + debug_suffix: "4220bc596083be0842144c6a014ca5e7dafbf21c5c5dd722db4ff7fec98970bd",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "23709e8103826b63a3d3368b8202a60a71f6ac477989fb59643106cfe7b01186",
    "sha256" + debug_suffix: "f31deadf281f041b0f47ba120f06d399a1fac78c5f629432a655c81d6a8cfe7d",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a136c3521163d749ad00c3ce0cf5a62649ecf4cf72ec5ec8e7e6b2c539e2613b",
    "sha256" + debug_suffix: "7c26c9762194ca65107a5351f51b20fd155cb4d5c65933c0e65a4c2bd0a20e43",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "8944feeb7b2e91dc78cf09b8a5b4ae6501753ebd2fb86bb66cc838e79e2dc066",
    "sha256" + debug_suffix: "a75c5ed8278ea7a7f4db05483943ed3f441aceccd801eabfe1a20fc90e9525b5",
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
    "sha256": "93ebc2d7fa7fde92629e807884d4499e4ece66784b4932f64ba8e42a9a9308fc",
    "sha256" + debug_suffix: "98e94f9e1bbc978477d6265578dea1134696a8a3da075a6e7e9472bb09aa59c6",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "c38cf5240893a4881ee3e0e9278ba297f2edef2ce3db9be9cb01f6c04c4b4c36",
    "sha256" + debug_suffix: "2fe8c208e1c588e3bf4c1c016687479b7b1d2f36c53ab2349333c9ecf3001b55",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "a2b3ad8b17d48c8ee071eac81490b2042581e4154dc8cc191078bee8585771fd",
    "sha256" + debug_suffix: "bdd06081dcbff96b5ff8d88fba8e3be7deaa13efa0c4a988c88608ededd4d521",
  ],
  "kernels_optimized": [
    "sha256": "369961c713f400e0de8f68445951d78995ce413d54e77af75892913ac1561728",
    "sha256" + debug_suffix: "15870bd13875e4eb72948a0bd18f13fa6cb087298f94ed913df454bf3261da04",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "c6460a5e70a4550e0badc14592b54e2065a65ab61e1a68631d65dc251ca4b007",
    "sha256" + debug_suffix: "8227ad8bc3aefdf64cadcdc340f63e85781701230ed0b5ea041a4ddc75273e12",
  ],
  "kernels_torchao": [
    "sha256": "b10ab7289d63f852adea7ed83d826d7d55cd2b371ab2994bd075f061020cb624",
    "sha256" + debug_suffix: "cca4a442474b799e1928899d43d27810caab6c3e70bf6f2ccda077dfd1178893",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "b0d55b0d0972698f3768caaf6f0aebea99a66aa6e68690fb7d592223a0fb0e45",
    "sha256" + debug_suffix: "6d9d9d73b69a5ecbbbbda4521d8f8666cd22962723910da16dc871e1c7207af5",
  ],
  "threadpool": [
    "sha256": "c9e4c1919b4727fee3ee094ec72249fd1fa24d9229b13e31acaf9b572acd42c4",
    "sha256" + debug_suffix: "53afaf325ba0685518874eaef99acdd069fcf11d74e097375ec79b6d7af94587",
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
