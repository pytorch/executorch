// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260204"
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
    "sha256": "b4c5f4372355a25bbae6c9fe80636a3120f650d16177ed197828e9a0921e7a2e",
    "sha256" + debug_suffix: "69a0526bf588a1a4bc6752b44040097ff2f96b52fac058316ce80df3147cad9f",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "126af7b66f120bcfa7539ff7058310066fa28beb8cdebc4649f139373daeea92",
    "sha256" + debug_suffix: "f8c5d70d35acea628dc60f7eae02b45d1ca9b87ebd91ce304cf256489d7954f2",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "55216aa2bc40667fcb7095fb97211c75ad73f8e17c0272614e43e4bad6e4d9fc",
    "sha256" + debug_suffix: "c18df63e00adda20e81b324b89dea932239f4cf8dd6aee1d5b977d6dcef4a6f0",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "45a91e2e550289947bb5167ccd5de12aa7aca735d52dcbe943b5943d1607b58b",
    "sha256" + debug_suffix: "a7bd6667f4e81b58940e38f38e4f71810ad43a67b852aba81cfeefc2ef8c4fcf",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "b3337eec2b453524bc637c0d6f6c947f1201f180506d19bee8cefe561d4245fe",
    "sha256" + debug_suffix: "27707243850e7cb28813490646c7a5dd876df5af16308c23153f6434c32e7434",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c306544c22531dc5a5c1fbbebd50f3ca117658fee777538930e3e2aa2c4285de",
    "sha256" + debug_suffix: "164d678cdef32ca48636f85e3193bfce1d08c1b249af1aec334663469ff6d272",
  ],
  "kernels_optimized": [
    "sha256": "0d1b741eed923ab82e7e0794e15ad084c23ab402de8f013adbc676a0315240df",
    "sha256" + debug_suffix: "0d4e21727e1917895bab9b084c4b58ee78d418c00ac0a38dbb8d6b5b5d6f7efc",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "efc89e962491689c3dd57c535d3bedb3c3a2c4feaf8591c52e3949500910d85e",
    "sha256" + debug_suffix: "48607ddfeb37a53d7c673d329c3da6c2b0b6f00b36f6eaf544aa4a49a8b9152b",
  ],
  "kernels_torchao": [
    "sha256": "1742d442bddb644281a21531827246772c22a8baf7959b55829e29d8b8d6b9e8",
    "sha256" + debug_suffix: "78e6dd4a333eb89b56f17a0e0d1c075ddaa30e21ad96e290eaf3b62b25e9b675",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "8782907356a1b6f9e55bf9efe97da7ae6932253e370cb421971438bbf45acbcd",
    "sha256" + debug_suffix: "256d90b91502631f0b534b9712aa333c5113e3a4878412f22f9798187db42bd2",
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
