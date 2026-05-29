// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260529"
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
    "sha256": "d028780e698a8613316b01abc9385461b7466ea17680630731f2b21fd5e7f112",
    "sha256" + debug_suffix: "d1805e56b58bca50e4e8a268f37afedc6fd011757aec4702956d90d8be71ad44",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "65498af162792f3a87e3b0ae42a69749e40c90a73de2d3cccf81d0cc3e54f875",
    "sha256" + debug_suffix: "c1aaa0f43849bda81af7893d77071edfa45d23f2f3b044bc04f9b6566e0b9de9",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "a4dae7909f25971a79c265624ccf9675f947529ba4201af16ec10d882a95e188",
    "sha256" + debug_suffix: "ab1d6686c14c509dc5cc6dc510af094ae1b28909852a3dcf194e15a38839093b",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "407a655881f395112c40508586ef3db2ee0d3614a82ce6459461354b8f31cb02",
    "sha256" + debug_suffix: "4b0f98fa867ba7439953d4e47736066ab2b1e657cd7c25665a784289f8db1375",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "59b7480835fbd73e38f6ed36b57feeaf774aad5a8fa148091e2b57c102dfc720",
    "sha256" + debug_suffix: "1bd0df885319dfa4f48311f6119d14b18de7cc48dbed1354701e3990df0be217",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "3fbb441a670a04af5f0120b3f2da8f2d5ee59ed455fcd98258728e1ef8e4e5bb",
    "sha256" + debug_suffix: "9dda8cde2c1abaf64acc9a55d493c9be318b766e588fdd78f72432da24212a14",
  ],
  "kernels_optimized": [
    "sha256": "205b235fb5ad6befc4af0e930c5a05bed7465293818a09be51f3571aaafb676e",
    "sha256" + debug_suffix: "ceb09829aba78a6307a19c8f24aaa33546cb7170f875a14c44cae4f4ca81d9d8",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "5ba465014f36f314b78cd9c8e07db225a1d83ee07b882e38898e4e28b6ee72ed",
    "sha256" + debug_suffix: "4881690e3e95c37050a25f7a7cbf8a9525cd7607d22d1d20dcb40ba3c1d219ec",
  ],
  "kernels_torchao": [
    "sha256": "9e99ad1b1114aa37f9961230a72531fb33d6e91316c1c35f5185b4b6028ad737",
    "sha256" + debug_suffix: "13b64ff43482ea6d5cad03688aa5bc04646144c0ba51b08e2faab6b632f2acd3",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "bfc7b2784f15f0d5d2641d185cb26179f2b774d3ef8d9301622668163200ed01",
    "sha256" + debug_suffix: "c59fefb169190242829dc79c4458088860ebf9bed84da4e4eb8fcd2f7b465f87",
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
