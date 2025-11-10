// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251110"
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
    "sha256": "ce02f74a4828675d8e814a280913eeef5955c950705655736519feac1a9a9cf9",
    "sha256" + debug_suffix: "50e62f29b0218665c4ad963708226c3641bc9de6d63d49e0925a66b57c9a5bed",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "3487c7d1dec542d265ebad3985dcaf408278cb28a14eef1c61ff9f9c7fb60b0e",
    "sha256" + debug_suffix: "3f9a1c93f2ae5a3888cbb64c48102ea968558e8fb5205b180e561bdedb143302",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "56b26e848d1e4699e0a32ad23221ad216feb74a58d84de5a300824999a9cb8b2",
    "sha256" + debug_suffix: "89c06f58b59b36d4cc7805317fb47ad6368431f6edcfd43a1f32dfa6ab4132ca",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "b849905cfed7ff48fbe5f506d01aa31353fa987df3ec43bfaf41cca9246e4395",
    "sha256" + debug_suffix: "4282bf154c1f9b3f123a468b74fb89b9e0dec88a7e3abdc2fe3b24ae97790067",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "c8293f3a6ca8b7911accdbbe241ff6b89b3f48e31b0019c4f4d06f57b41b1f96",
    "sha256" + debug_suffix: "4a3685fed4268f910304e9e848e814df94435ee64681914d9669e7b217f301c2",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "ad8ec9475b9fb8a6694a6fb4f0fa2a6b9a93d0d2724d28df663a215aa7a4542a",
    "sha256" + debug_suffix: "88314ca6ff37a6026f3ada3e5a89991a51520f3c71ac45d7c3529493840b9bf3",
  ],
  "kernels_optimized": [
    "sha256": "ec08579acbce2f0d8d6437391b1f8b5a08b594135498836bf871dcd90f983837",
    "sha256" + debug_suffix: "ef194282a245c07d2931f1b04642b4cb48ed232949fe784372d673dacc08a85e",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ed39c4528f0c693ab0db6a49f76b4b1095ad7f525293e3cf08a664a5ece7e1be",
    "sha256" + debug_suffix: "416b190ab91e02fd57e39ea5f885e09e05141dfb979532294aa0c3075588ed01",
  ],
  "kernels_torchao": [
    "sha256": "ddaaa52b02c191de8d7d306d44ba2582e1bbf2a8d4b59334e62115714971408e",
    "sha256" + debug_suffix: "134f44211babf908574cade92fa114bda367fbb6f3a53ef8ad0354430fd2a301",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "b230b91b74a8deadfe08b39e3847560557e1867a7659ae97140cee298b5f54f2",
    "sha256" + debug_suffix: "9548d84b1554fd248760ac4f08a1f84da0fe8c951dec6043d6fcbbb7269e830b",
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
