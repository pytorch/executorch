// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250913"
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
    "sha256": "b12aeae36d5eac057056b91706b8e27f5e498620505f89c44ca9d6dedc8d52f7",
    "sha256" + debug_suffix: "b8c23cd5607a46146ed5ba1053afbe7d57d728bb683ae3eacc1226b76c4b965e",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "aa9607c59c47fe6f3d193cb83c90f4b884927d7ab2e717c283599c3b2f69e716",
    "sha256" + debug_suffix: "83b3b04924002ea86728c247c06f1f604a1b6f8f25c593ac8676183368d77d53",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "1f1ecf3eb4de7bc84f4f0cda2694c73a595e4846cc5e0e873ceed80eb66a6655",
    "sha256" + debug_suffix: "030137f424a173f70a4a9a1641001c1029e12ba91e0b05b477372fa564d841e8",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "bd04f907e22f4992cef45060486813a6050fad1681a5ed23b7af37f272732855",
    "sha256" + debug_suffix: "3e8cef7f3e476daf8cc1c259703e26e851aad9a290eccbd109d9961c7d554cf9",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "95d3fbcc58ced9c797e2162776a0981015e6ce76b4a62cda7a4e1629790e87dc",
    "sha256" + debug_suffix: "f8c398bb224eeabc5defede497dc04f6dd043c064577d6647898ef5a752648dc",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "c8942050c9fa4091a9cbfe75e74916be6c1209f71466d81645c91e9c05b965f5",
    "sha256" + debug_suffix: "d08117c7f5625bbbc16ba8bc65cb2690efeae4896386f07c8052798508710c67",
  ],
  "kernels_optimized": [
    "sha256": "b989087478ec810e457513be3e5a62052ee5bd80315ac617880e11bbc0c70aab",
    "sha256" + debug_suffix: "4d9edebd23127b9e5e4f97ac0ae6a97ff83ebc88af2abcd1656841c0756fed65",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "aabdfdab3481b4ed15ad253831dab36463d9310ec476c3268ab522d6da138a78",
    "sha256" + debug_suffix: "3883b17c7f3400e48e0a33d830be6811b7270eacc05eb1f38e3e511e05892235",
  ],
  "kernels_torchao": [
    "sha256": "29d14758a6dac81e9a1bccf83c042aa0037ef1aecce5b3c42f5d06af9b01ca91",
    "sha256" + debug_suffix: "8ba8d475864883d461fc789c8c9861ffc5229448b2fa51ac76100b6d2fc6186f",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "f2b77d089bc10d6a553b485fb19a3236f62ce9961e7cca7c362b21d0d4459ec0",
    "sha256" + debug_suffix: "d9569df1d51cbd58b9daf3575561348953044f9f0ea216a91d78600f5bf94f9d",
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
