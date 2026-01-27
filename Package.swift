// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20260127"
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
    "sha256": "8bf4715433970821a97dc523b47646449af0d72def441dae7cc2475dc9b8f483",
    "sha256" + debug_suffix: "4fa1d4e8d6729b7616a80a7476be5caf4d897af988783383b06f0121d99d3495",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "e134a903110072a239d3a7197ba46b2e541ebeb693db0331e9a7c075c5c6c9c8",
    "sha256" + debug_suffix: "563b0e9f4258053d279e33bca476d4c9d095ff5691532c6f68baab5ff725d845",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "467151bc0e56e1f33935097cbf26f9e1cae0d8f2aebbf9fc318fdd2c24d7b6c6",
    "sha256" + debug_suffix: "3d5cb0ba4e7a420f37970e524c030d2335cb11fd8eecae70a6994e93686ab814",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "3fbe8779dae9f1c6ab6b522755ee0485dd08dd586d3e0fdbaf7bdc06e2733088",
    "sha256" + debug_suffix: "8f9b0a419068dc673d744182f2cc5b9f7743fe3c87aa46a84fbcce761cf79810",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "131a4f8ffe1549a54fadc16ca615511ad1dc217d4dd81ba5dc93b45f977684b3",
    "sha256" + debug_suffix: "0a01a2468b74a983ffc5c63431f0c601f919489d32df969587cbef8b080b5b4d",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "21c4e5d602db493ce011013032d1d6d8a1d367e6ed48db95b977caa9909fee19",
    "sha256" + debug_suffix: "c1078c93b5f365a02179db464ebebf8927a9b8ed72f85716cc60efdc4a1f7fc4",
  ],
  "kernels_optimized": [
    "sha256": "ed5c1c57a515cdc9bb74d8b3a5873784621335875335b607dcedc7179868d495",
    "sha256" + debug_suffix: "428d37b52b01f1768fcfb6366f50f6141250f541bb1b8c9358981e52cfe2911b",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "128a9cb16bb1b1ac289050d54aec2881f5a6af05c67e1a5718e93fe8fe2e174f",
    "sha256" + debug_suffix: "bb6fda74ccdfedf4c66119d40b84f214ff4c94b897eb194916b1096d03443f22",
  ],
  "kernels_torchao": [
    "sha256": "daa824606728578769c4e8a094e74ebc5b6ea47fe24d75c82bbb3be0dcb2ce38",
    "sha256" + debug_suffix: "a17dbffa0a3511f47cbe376a44a6e0b78ba1066f52b3fc78e8a0c4f1f36ec25a",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "fb710bed7971846ffe1a1b5a82f5114fe9640c8ede815ae138c30589328c6065",
    "sha256" + debug_suffix: "5b1cfe96d52d24e1c25307e5d8aa76cd9c95982eb98d4b5bbbffe72fbae0732f",
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
