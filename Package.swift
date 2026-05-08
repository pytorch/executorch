// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260508"
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
    "sha256": "1bfd99336baee7b74781bb815ad199061db1c7a20fdc39e2b5fef03afcf04b43",
    "sha256" + debug_suffix: "5c5a3e8af52f47321b99d43cfac03d27185904c2daa52351396e38d8e918d762",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "ddc8e849a1487977a56dfaa66fd028b3b12caa5ec1b97cf16bf7ff395f79ae0a",
    "sha256" + debug_suffix: "716dc0a04b5cd128bbff847ed6b33dd321b9f2e766b5dd3d2dbc87be2105772b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ea8a3bb020bde6c69798cf8e094b210384a7d24c0b6b833b86921014a501fa73",
    "sha256" + debug_suffix: "c4b0e6a121564fa6ddbc980ccdffbe8f3ac0e5ffea730a9665225e29980fbfca",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "6b3dcaffc3f770942532b0d15102b77c4ea2d190ec183ce070de850662b65622",
    "sha256" + debug_suffix: "3c1037e2b8a64e180488d97d6eeaae3720f85aa6f04779d31fa544040d794f9a",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "09b21dc24db0827d80a3ad2f516957833832b1c18752a910e42a9c86554bf6c2",
    "sha256" + debug_suffix: "323fe2294163f35dd14f6d1f415ce4b7785e4fe1d882b418811fa81ee17ff644",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "9eb1e2eba8e9b493b868ec868b07114ace240edca9265cd3c1f62ccc3ba0dad3",
    "sha256" + debug_suffix: "47e3612c12d3b8c76d3add212913d0be3f2f5721fc478089e9da0092957aafac",
  ],
  "kernels_optimized": [
    "sha256": "c3f09b40a0198300024e59bc073c2d374bd7727c62e8c13730b9391a0b048ee8",
    "sha256" + debug_suffix: "f7794f4de46d5ef6076f8486590c241213706210e8b9e46f5abe8eb9a4deda58",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "c1848e60fecd94c7b76acd4efa0263e9c9ca6800aea4c24706014504dd83b9de",
    "sha256" + debug_suffix: "e793d23b221b136f5088883ea974d1c93b9f74d3d8adc2bd9c1ac68464faab21",
  ],
  "kernels_torchao": [
    "sha256": "cc68513f2b2c8753d404424925335b1be96c2bb6761c5546179044e4a27f7cf2",
    "sha256" + debug_suffix: "fe10863b28c5a5d670604c518e48bda87b016f3d650264443637750013073a5b",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "bab409dc1ac7e896576ad21c3aaa060eb3046ad534be31845cfc7bffe3e0dacc",
    "sha256" + debug_suffix: "edfdfe652dd5c594f3aab69c55577b72b9e395f54ba110ed9d923679f0d599ae",
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
