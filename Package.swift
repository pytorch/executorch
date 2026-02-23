// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260223"
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
    "sha256": "fd821af27e9ee68955b4954713d0ea25801f942e95e6de024082ca4883e2c0b0",
    "sha256" + debug_suffix: "a45ac9618ebec12b618ba91709f393b89667e9bf94745bcd6fab36f38e3a2663",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5dfb2c7561ce6ba2568dc4e1283be4f79382fffe647d0362b9f57261f774ee89",
    "sha256" + debug_suffix: "d4f01a71fba21ff5d0ef5d6c2fb4c21b29d91e072a062176851cd59b8cee6bd8",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "7bd0ec82e3815976b51bf6870b3b7a41740c0a94f24ce1ebd66d406339903342",
    "sha256" + debug_suffix: "89e280eba552a924a679f4f00d08e76d2b905e9000e66674cb50a8105b610059",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "598030d4e03f69d1110bf508d3e8197ec72df81fd79dd569120da2d34bc2ba4c",
    "sha256" + debug_suffix: "5059e1dfd57979efa0e6777875ae298823d184aacf4615eb655e2efa124ee108",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "ba0e5b5c7c7fda05718db5c814fe01006f70338adedbb2bd9854ddaf11d24f0b",
    "sha256" + debug_suffix: "1f0ee63100ea31724fd083032a38101e53ca332c9b1b182d0720c69c0d545a93",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "f6cb3c33c38b1e0feeeca147e80f8219e58020d89295000dc3f67df7b856d814",
    "sha256" + debug_suffix: "9980370686c68d333f9327d7273735c3bc0358993c45c131136433d1f1fff00f",
  ],
  "kernels_optimized": [
    "sha256": "1dffa83718ddb7530fb3b3886a658f20f9bed0d6bf0373e4167a050ee6f3067e",
    "sha256" + debug_suffix: "b424215b9bac6e2044e72cf5fdac32a61836f894eb29d067eb610bb7ba2dffca",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "e2a7973402d06cf287830fd096321d161cbdc9b882ddb10756452008a073f62e",
    "sha256" + debug_suffix: "ab69c386672511ae3f84917d739bcf2a36f4026ad7bcc239ccd2d8a516a441ea",
  ],
  "kernels_torchao": [
    "sha256": "143d2ac3dca369fb2da28c5d1561da5415747cd1fc773cadbd83c6e4cc2465d9",
    "sha256" + debug_suffix: "a40ce9e6f74ff01549ad07df1d3651754f7da09413b2566bed954edfa10c9db3",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "19fe857bed5a90470c7a9df3b2fd84f731f00e366a5dc830cfecb2ed8f007c17",
    "sha256" + debug_suffix: "64bea0c2660d898cbf9e10c44d1f9882377d2ae867185ecbd9a2f331b41f4115",
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
