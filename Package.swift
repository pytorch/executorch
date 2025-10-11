// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20251011"
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
    "sha256": "2528a6626a98ae56626f67573f1f5487b679f363cd30e0bda25a0475e569429c",
    "sha256" + debug_suffix: "0b196dc73467c0f96190a25c94df43d71c3c34b7dda83721dd483f45a5590740",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "32f6f6170b5d6fb5f63408f2727f4178c02fb8b140eb4782dad05614768d5160",
    "sha256" + debug_suffix: "cbdfabd466ed123b49c93b47c11600897a36bf7eed20a3a6d51b7e488469ad1c",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "cd15ca5e06b1270c357561cbb4f7034c7e194a825fc35b9f9c1deaa1bc3e91ed",
    "sha256" + debug_suffix: "ed417756ed9702bdde8ebb76ed91640db0e5b6f1064f7938d27005f5c9532007",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "bcd336ab6bf97ea75abdb1cb0c609b6e8306c1025625e859bfc1f2845e5938c7",
    "sha256" + debug_suffix: "a5b14ca1dd36efbd05d45811c258952bfa38024dcbea22341e24903a29790006",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "d6c15168b3e5834d3f52fe64e92f1cb5d5e2fc3f0b1b8010462228e9433f6ba9",
    "sha256" + debug_suffix: "0e938c758576277317ddf09bdb2c1713040d7bbb3f55ec44f98fb18754e43492",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "1365a1c0c45420ea49b8398bc06ecfaa424e9aac301205ec431c77dbca5f6f8f",
    "sha256" + debug_suffix: "43cc0b38b55599cc4a63f6480cbe3c81c9e2810b57e46a4ce5365b7c7b37b520",
  ],
  "kernels_optimized": [
    "sha256": "e13b5fd69f1f8341d4aae9502ce56c4768e1dd9b24a18656bb7d3c27a10e738c",
    "sha256" + debug_suffix: "a9ca10592a22bbd8037f2fc22c2368e3279cf15094ae44692235e23e0e76a786",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "d7535df6f22cfeafed74c282e93718a1538bd1d46abfc3e0fcdfae1b07a41e3e",
    "sha256" + debug_suffix: "6c687a830eab9ad0a1388de3d92548e9e4ff866b59ca500365a09fe86cc7f796",
  ],
  "kernels_torchao": [
    "sha256": "19b838db6ca1f2aa514b8189cd0cce1aa64d04d39377569d3a1053f836c79a28",
    "sha256" + debug_suffix: "6eeb6d728ccc345936f6d0e4af0713075c7d6c02429d5145a005181ac949ac03",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e874d96781295d29ae5e3cf6d0646386955eab87373d9ed739aa56a7af902d1b",
    "sha256" + debug_suffix: "e1294e0b964b2d6638bf7b683bca254455390d01b9701e8d8a51373275c79664",
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
