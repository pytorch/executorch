// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.5.0.20260903"
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
    "sha256": "975d7c91d506096f2287e0b9594976169503065e3230d73369de4a726a4563c7",
    "sha256" + debug_suffix: "bbc61af9bc3abfc08b864cd85d9aafc51810e6e93503b43fbf5a79f957f59403",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mlx": [
    "sha256": "1891c10ee928ec5d0fe438a42c0591c9437fde4337175a301fdca41b6db3c0c2",
    "sha256" + debug_suffix: "6977b7c7a31d6429732f54f9cd5d33d44697eed54fb61e89bbdd625c913ba2a5",
    "frameworks": [
      "Metal",
      "Foundation",
      "QuartzCore",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8688819fa1be970ba03a8eae480cb255feb253ff87705286e926d13a5e497cc5",
    "sha256" + debug_suffix: "f13edebd2b9028711f2d97e8a24c381c5cd973657bccc0fd27fdb41959f14b20",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "0ff966c322e2de18eef21e4e2dbe17cffa3b0c39b61c30b1e7c6a56c268d8f1b",
    "sha256" + debug_suffix: "032e66d693850c110831b99841d5267a7fa9b88a6578a75f5c6d77074c4c5a41",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_dump": [
    "sha256": "37ce357e4cdc4421a56d75898a39fe3fe1c2bc774d7e8b4c6b54667ea3885f26",
    "sha256" + debug_suffix: "16fe742363e17b4b0152d131de4c1e6bcea52e50e7d2bb1125984f80084fe4d9",
    "targets": [
      "executorch",
    ],
  ],
  "executorch_llm": [
    "sha256": "79bfc03dc1c15ebc9cb6c7935b1b9859e98f4812088563e83344e198277ddc90",
    "sha256" + debug_suffix: "a6065ae4bc735984c87cc1474493b011553523eab93513809c4a595e8ebb8042",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "920b91847e37c4e32f59e300b49be1d31c022509bbbf407f93e336a4e3921937",
    "sha256" + debug_suffix: "35b67c64361da807102571d0f43a7734982b1cf87d7a4101989e606247ad608f",
  ],
  "kernels_optimized": [
    "sha256": "6c2fd773417a24beffa6071cefad34a7620bae775657e8f8f902c5561f3cefba",
    "sha256" + debug_suffix: "a99f9d64a597231be4b3b26c8a430d8124e3a626b551f28d8d109ad21a309456",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "13f9144ce6a4c98b8914cc0d2fa41ffa13ae49bbcae98d0d56626a8458b3ed85",
    "sha256" + debug_suffix: "81e7cc811ecfea057c33e1ca5e7597f574911389f4bed08af1c85fca2e468c5d",
  ],
  "kernels_torchao": [
    "sha256": "f8f105af8e53826664f2b95a9c499d3f6b0e77c39a29a807dd0537b6564c91d1",
    "sha256" + debug_suffix: "e50b53b3bca53345eab3e0210b9aef2e28289ef32dd64350766a260423c6d52a",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "3be56193078ce00e9802016d3e843f5367b0b50f9153e802d1ffb2cc99ea08bc",
    "sha256" + debug_suffix: "43ee4e9b4256e07b6a50b35adf56289384162d702b409c48759b43220cb8b17c",
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
