// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.6.0.20261102"
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
    "sha256": "df373076c3f8d74794af5c9d40e0f84bf09ce4473b4bacf19d6d82bd0959eb71",
    "sha256" + debug_suffix: "7202e1965a85ff505e59e53eebfa75c32716510e97695d2bf39a27f729635cad",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "87bdd471bb726678486eb5b61fb32dfda81cf12245617f76b04aca0f1de167bf",
    "sha256" + debug_suffix: "a06e1517b4b20d345f12e38f660eacc37c556d1c67be27769cd4816df4bf3086",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "9e10e1e1182da1591cbf21b4ad35e6a987a6df10085d491b78c80c5a2f146a41",
    "sha256" + debug_suffix: "cf75fa02ad9fbeb4f79f8964668758cd454e9f58b8dc7a253d7c76a184dedf92",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "44e527c39bb3891a0d3cbde320e4967b46e650da0078fa9607a90a8cdd125967",
    "sha256" + debug_suffix: "90659e392c7d3f9145d9abb5f017e408f6efd427cd0ec0b62c8a54f53ba558b7",
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
    "sha256": "e1a24e085cf1562a383286ea14bbf36d253e5260490c360e86ca96871a3ada3f",
    "sha256" + debug_suffix: "7fac5dfba2073b124cadae2b99e5e56d3a9819056f210a2015a58e3d608c2f82",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "ef29dd9b313625ba4e8eada4ccd4facc86c53520720952dd822b7782404aab2b",
    "sha256" + debug_suffix: "3a8833e18ae60b18fa5e5811c1bd7ff5f233607285873a7f4a977fc8c145f354",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "496966df5bfff4e15670fb311d7dd4465ae10a656e8f4b9b94fe5840e801bea5",
    "sha256" + debug_suffix: "5ad228e3e3f14f7a8e33648fbdc9b875be5eb1e02bf0e0ebf0c2d4b72c2ed34a",
  ],
  "kernels_optimized": [
    "sha256": "357e6351295bb36bc3080609c8ba8cac06801d6fbe28524602718893a3c8e45e",
    "sha256" + debug_suffix: "f920824f8708c284293c65315733cc21944bf5e663c8b7e29d5b67de72c86d54",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "1007af711beea197ef707f3193613b973198391dfef3781ecf8bc6eb774385fc",
    "sha256" + debug_suffix: "e3450a2288deb9852796df62d64906d1b95cf8e6b1bc6ee42aa382699e851e63",
  ],
  "kernels_torchao": [
    "sha256": "81f4fca136b59063fa6b823a5182f7634d6e5644777ba5d91f2a355232c87ac3",
    "sha256" + debug_suffix: "bb3ab5dc9937e58120e6d16773603aa2eb52b8b14c262e630f541b8136c8f100",
    "targets": [
      "kleidiai",
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "kleidiai": [
    "sha256": "a3f12615a0463cca91631b3ff198390f4f69d500578c38d899e582104828594a",
    "sha256" + debug_suffix: "f35ab9c9aad08740ef5a0a42bebce44aaf82b22db6279d65b8dac54c4816a71e",
  ],
  "threadpool": [
    "sha256": "aa4bc1aab0219274c341c08daa50acd26cc1f2d12258244145d57ad0497b3f41",
    "sha256" + debug_suffix: "814ecb2103c35c28058863acf0d4694b8538b51702068ef70eb4a3f6f103b6d8",
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
