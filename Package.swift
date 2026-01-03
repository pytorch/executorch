// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260103"
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
    "sha256": "5c684afc6bf8a381e959b4a895c91f656514412a2fc97ee5856594eea9e319b9",
    "sha256" + debug_suffix: "642ca36702bca0687ba49467e77b0a46340125518cf03f076713494ed4966986",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "cd286d1ac05f1587a904b680cd07e1e792f3042bb26bf0621fdabbda978306bf",
    "sha256" + debug_suffix: "dcf96b2f31d795a4c9423fe7c5a7c2d784ee4311f231357572341de8776979fe",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e266140f8fd08f53cdc684d9b9712728ff1ce2d5dad319966b6e2c1515a015e6",
    "sha256" + debug_suffix: "a668c91eb6b12ebe12ada5b43f21b6f669d8c206611260e0051360bfb6cd08fe",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "7d41bbda67069a906e5162fdedfc5d1fec7dc42a4d99f4ca2f7046c154de4e10",
    "sha256" + debug_suffix: "791153180cf6809f4dbb27dc109c45bfae83068a6dcd79e77eac146ff5a1ebbc",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "ec2e5e02a32e9af976ec0998cd5b2954be35b70667aad88cef70b61c1782b68e",
    "sha256" + debug_suffix: "c18f64d0a101e0c28cd474e8fb9a8f722a5c753f8f4adef46c1bf8975269d250",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "e09b2892d8e4866918121afcd58ff342220d570578cadacdc88bb7b2c01080e1",
    "sha256" + debug_suffix: "53ae5601a60322717cea5703ef22295d395e5496d74e29fbe1fc954b23b42216",
  ],
  "kernels_optimized": [
    "sha256": "98b00451fa19628dc640fca0fc5e4dae5aa16c3f048de37ea3cadcbb48576ce6",
    "sha256" + debug_suffix: "3c18bce79f9b30795e63c7152e54445aec16cb33368fc40884b461e7d43f8cfd",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "73a82567c9a83acd089c6463b50f9b1b5f1e6b075500c4c9dcca0811491d9f3c",
    "sha256" + debug_suffix: "4d7f89162cd0732022cd707919828df516b6b23f87d92afe7bf712d9e9af391c",
  ],
  "kernels_torchao": [
    "sha256": "3b882d1e879721fd9539ba30176c50f1a77f0c6c573fd2290ee7e49e92acd1c1",
    "sha256" + debug_suffix: "ec53e90ecffa21f5b4b26faa68c1aec9dfa4530f45b3bd9edd95e0d9fe3d7c20",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "5ffd1c11674c09edbfcef1b8fb24679cd20539ddb3e80f2286167748905e4f50",
    "sha256" + debug_suffix: "74461ab66801cd154ee5fdff06c56313e35ac6e9e6a7ad6a175cce3cfc0c982f",
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
