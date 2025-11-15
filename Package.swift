// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251115"
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
    "sha256": "645e14d3e1a73989bf1701113aee503c7585a5c9fb0d67e70ab13a06fe299810",
    "sha256" + debug_suffix: "c7be034211d3c8bbb9382c13a2374ecb4ebc2ad46662f3a0073aa46bcb381d3d",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "3acfacc03681db2b7c39b8ab3eed7e80b0bdbe60a3b5eca5b5db2be902a2d0b8",
    "sha256" + debug_suffix: "f14cb350887b1fcbfe556263e96dafd37f94a289dfbb1aef6a720e176e093b6b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "83ff22ffc39c04c5958215665ba66d50635a077d7e987fea8959adbeabe7af4a",
    "sha256" + debug_suffix: "027f784a66552a482fd2bd1228586fca9b1d549d34bbb3757eded51436405190",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "eb4fea838e3970f75f041679bff9de4332d4700ec370dc27c7fda95c84efc06d",
    "sha256" + debug_suffix: "54286d38b97a2e6bdf429293ce946d58b1500262743106d30a3ba392bd3520fc",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "1e42fffa9c48bdcaf8107564db51ab532148272eab43bff83564a491fb007518",
    "sha256" + debug_suffix: "7cce6ac620cf9118706f574c66a4df77c5b8457fd842f8dd193fed085f7be255",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "15f34fe6b5890a2bbea0b04c08665b398d0de580c8e9115ce74a8dae9f666a75",
    "sha256" + debug_suffix: "3330978a4343b9937f2899028709c7b2ddb0c853849e7054c1247ca3630f24b8",
  ],
  "kernels_optimized": [
    "sha256": "4bd1a3bac4fb9ca374ead836e4113be009bb5159c083f43f94d93b45ec74ebda",
    "sha256" + debug_suffix: "eaf5dba1deb56c7a24b5c0bdee636b5d0f35437528eac5467796450a168cfd0d",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "1e6e8a2f00f16bb7b4391a42da3577f1f8b32e22f1c12d3fe9ff8f7121e9c06b",
    "sha256" + debug_suffix: "4cc5e4d4092044640649d7ba99d7339fd3de4c52a321c9c511986378836f4741",
  ],
  "kernels_torchao": [
    "sha256": "740fa1d3537cd6e1ae2b465bc38a4440ccdaf9a610b5e4704191b078a52b7db2",
    "sha256" + debug_suffix: "28feeef580c69ebe1aec402ae3ab657667e52152d4b22676827021c62f0349ea",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "66c427b70c99ef0d0b038f7698c0944f19c0778aeca1a2ba93552eafdbdb4c2f",
    "sha256" + debug_suffix: "e5771e0e86f6924be50bfed716a6213d0db5a1827a0153d55da937c782a62f1d",
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
