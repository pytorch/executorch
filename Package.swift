// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251221"
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
    "sha256": "a6a9ef6c8d77ed770943ec9160a9c257ad94e3176a76fd88b270630b2633bb43",
    "sha256" + debug_suffix: "9095d6d19290e078e4f2e1a7bd3dfd60500efa6027c4e2b2718231c237441e64",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "d8812b29096ead51453dba3590bf60fe8ce383bc9caf90a3a4d823870b8f7c39",
    "sha256" + debug_suffix: "d4d4c32b3bccfc48b05b804a38b53c0390e1943ea3466bec401adb903c0ab878",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "e6600311641cc57d540932cd074e1118c2edf2851384067908f372407f51568b",
    "sha256" + debug_suffix: "2eabe1a3aa9622451159ad86f5e1070bca4df5f26f4d4bd5a58f51e87279b430",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "19fe5c589cdc98ea2b95e11b7b9eed49828829f81c168a06c1adf5476abb76f5",
    "sha256" + debug_suffix: "0daf21296c586fd89c3be2cbbda8aea2c7ad3df944ceb88875e73e40c4b6619c",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "e082836dbc468a83726d5fcf92f43321087dfeecb516ef9abf77470eb2de8fcd",
    "sha256" + debug_suffix: "d82ef3200688b8000cf777fc94a9c43738c533cc5f2686d8640221c79e9e3d46",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "119931cb44c7541746e6cb156a9dcd4c214ef433e5af98751f10f4a6b1f3bff3",
    "sha256" + debug_suffix: "9a4250205b2c0651394518cffd0fc21a461b27781017a981d8b11c40d49b0a42",
  ],
  "kernels_optimized": [
    "sha256": "930fac19a8f2056a281f267ef0e40c1f15efc2589fa1ef145833d0a219101cbe",
    "sha256" + debug_suffix: "691d7e59ad8634ad6e1a1617fcfaf474c9c4934a840695e8078e67a750258a42",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "2b7991483da813ddc5561c4fa9ccfe6dcc7e4e4798013bfea3ed2cdea75bd2bd",
    "sha256" + debug_suffix: "f87763e6add34b66157acc302f1e9f327d39fd16c884c6326005570180d3dad7",
  ],
  "kernels_torchao": [
    "sha256": "fe607ac0557c94d8e860659415774772582b7e33eca23c97981dd896e07df6a0",
    "sha256" + debug_suffix: "818ff0cbad46639c29a4ea24a78c648ec773bc98613f99559509cf199c7f8b80",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "2bbd1aec433146143ceabc75800a120dfe87a24f060fd67e0e6ecea5c57eca5b",
    "sha256" + debug_suffix: "bb352e5645aed5f0e0ba9b8a1c4496350c617b011bae078e67f7657e9c4e82c0",
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
