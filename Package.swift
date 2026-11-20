// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261120"
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
    "sha256": "6598261e905fbc6ad28ac7f9e5b50a829611c9ecaf9f6466ad3a96834c1cae89",
    "sha256" + debug_suffix: "3e6e671fc4bb49a2db6a5a278d2a76b1feebb9470808086512f6a96bf463eb87",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "a6ae133f7b1468feb198b2869ece3268c44c2ad88172afe468b7c1253d9be8c4",
    "sha256" + debug_suffix: "ce5e971e697239df5d1644fc19c243f5287bbaf953982c3a3efb082074d179f2",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "0a850deca255957881e751d02f7f1609ad9216f8d79c73050cfe6a920ee6e5ca",
    "sha256" + debug_suffix: "60ebae8ff45fd000ad62cfb2982eacbc5a31f79615acdffdc88be39670e68caa",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "f077986090171437a0a9096c395e57e23cde3b6932170e16b19112e3e41f2f6a",
    "sha256" + debug_suffix: "114e363581d5445f1a747c025e5c2ea988488dfa0c849227f0b782baddde4e46",
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
    "sha256": "09f8573f8416569b3abf6b9487b81c7ccaa2ef2976f705fc8868e10446c81f63",
    "sha256" + debug_suffix: "6af5ece0d8dbe0293991960e1d77b26b2852d463cd2568f0bdbcf8e2f104b7f5",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "7eb446c14b0957657b07b2609dbf37d7f3ec25b8eb1127f8d664fa40cfa02904",
    "sha256" + debug_suffix: "3e298917dd4f679d735d8578d0322b893956ce24dcad22c9c1273efbdc6ed782",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "1eee80e2fc88f6acefffb78b723b2e9950848283beb842935d65c101e9f4f0bb",
    "sha256" + debug_suffix: "2d1fdb33f37e4a5fcc133c5759d9d8f592ce70d916655d2bbb1134d401c621a0",
  ],
  "kernels_optimized": [
    "sha256": "6b95041b0995a3e7c4431d54ec9b8ea45bc473e389db771b8870f14148f5b124",
    "sha256" + debug_suffix: "260f84662d3ab71034cc8a4fa57c46846b0086b2726b587399b2a1e06557e646",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "78570b94876724dd4f49191c4221c2b264dd2983ba53eac81c9f073cf6b4a149",
    "sha256" + debug_suffix: "933d2eebf2deaf7f6b308a1e8ab8df70b5184aaba0fd0567921cf7e4466a4d0a",
  ],
  "kernels_torchao": [
    "sha256": "8d66ee0b9d8a49a1cedc027054c48b798fb59520d0244cb29061dee2305dabcf",
    "sha256" + debug_suffix: "d2cd03ab4f550aafb56f17aa0c79a42b94dd5792ba209b564dbfa698a5f9d7c3",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "184c018e381ef5bf8b8eaa49ca51a03566649a6fceab6c05e684cc31b01f917f",
    "sha256" + debug_suffix: "80278e9c18220fd01796810118be545b4c0b22a662a61ef54dc638f8cd775304",
  ],
  "threadpool": [
    "sha256": "b27b4bcf17407f99d5e893dbd4d288b184b5e4c14dc7271c93a6cd83b705676c",
    "sha256" + debug_suffix: "840b21f962e0a34ce1c6c6ef2a325e9664da9558cd6ea217c1c854133c00d92a",
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
