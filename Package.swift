// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250924"
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
    "sha256": "fe78ca52de8ccb849be88baa6f4fb90a48510b4eace317a923eab53da3ddd23e",
    "sha256" + debug_suffix: "cfefcb25352f09ddc2824188c05ea382504a4d21c72c0964a682a3b07a608b54",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "55aef92119b142f3343d8dad66a14c666734db63920899e9e3c83d6bb963aaa7",
    "sha256" + debug_suffix: "9809388364de7578caa9417b3a00ee5fdc99b5efddd56c7cd04cf6687b6e3fb0",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "02d61f8ef49bcf6f6c3bd4459a34d6f2b77090ed877156e51e258c2ee0476b30",
    "sha256" + debug_suffix: "9a73a2f1b84c6dcb61a353539b207e6de7271fbaa23d53ac5919857d0b6fefb7",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "4fee30998d8059cea68d2b5956842fbd05fdda8439fd4403039e7c52abbe01e5",
    "sha256" + debug_suffix: "1da9b236a0d86d80455b32edba7aa7c64ab4fb30876ff99c2402372b852fd50e",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "653e78798f7d41766ccb07fc5c8c49d6f54d060bdc040cf61ed66ffcc633d7cb",
    "sha256" + debug_suffix: "1d37dc416cb6839bacdb8cdee65d675aa8b98016c2d488fbb98fc0c8259365dc",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "b41955287d3f47ff5563aabdef72326be224625ce429cda4043d4b9d785ca87b",
    "sha256" + debug_suffix: "a96263e9556f40e64418993d16c798017cd13bf7b446983df2e7df4b6c3a429b",
  ],
  "kernels_optimized": [
    "sha256": "893fb7e17cd3b85b0619beebe63ff1e6c9432301ecea7b38057afb7fe4d3cf7f",
    "sha256" + debug_suffix: "2b88fde637d1c7fff41775a16319f367e3c73896cf5ac61b5065f056ad037bd6",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "75357e44cbccac3e737f4c30ce0016c3c061fbe0f4ee4f5378b04eb54278663b",
    "sha256" + debug_suffix: "6918f5d5ad261e0e9ed420f5a3bafbb5e370a2c72544bac0108c72446e15368b",
  ],
  "kernels_torchao": [
    "sha256": "eb9bac78e6579049996d4a0699a47453dbc34fe569fbb8a71fbc0b5c8261a157",
    "sha256" + debug_suffix: "eb22fa98b096df926d36a0cf023b4fcb1f7e8e227b4dc0debce8cf632dd0d39a",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "8ff07a283b39728357e7af355af37d350912606c557d72d664cb26eba981cc61",
    "sha256" + debug_suffix: "013e6891f7ee0bc327a8746c8c718b59e670dd3efad0c751a54d56e68b8340c2",
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

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v12),
  ],
  products: packageProducts,
  targets: packageTargets
)
