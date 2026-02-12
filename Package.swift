// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260212"
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
    "sha256": "75fe6e8c2aea2f00ee5a1d0ca05e06332248bc33770d9e7faf50ba985f06b4b7",
    "sha256" + debug_suffix: "78a63fd8e259f14cce4cd8a85167d3669ec5dfe1f8eebd71e90f81729ef361c5",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7a627cd6d24dd5932d203fa062cd7829abde3a72697d9d493d309ada7a1a4052",
    "sha256" + debug_suffix: "e0923b0f60fb1191f5a0f9cacfcee4426859c1da5d7949c1c3afff746f264846",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "bb32043da1dc1820a9d507f3d75c84774fe3e8c4f5da4f76bad963d7a5385fcb",
    "sha256" + debug_suffix: "204ccec94d4f07ee34943a4b38f2b7489b1ecd08a6e85235a4584b502f104a9f",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "c879b47012551b229be523afda8f8b421c60cbe0e637059e9c3021962aaa5ea3",
    "sha256" + debug_suffix: "f541ae90a9217222953d15e0279edb1a036ad910f630df51bd4f8ac9ada1ff19",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "ae2e9ec66d5c52eb5bcf25c0b984ffdbc7dba2cf2aba5789a65c8434b8027b32",
    "sha256" + debug_suffix: "8a036a634db4a7304798f61a36734c1d178b1c270eb523e3bb4748182b000802",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "a95d2b30085d97b05ee018dfde070392b296c4f47da52d33a21498b7ef4fc272",
    "sha256" + debug_suffix: "ab07c5af910546bd28d3c77d67e4a1df4c49299eff31867e46f83fe638085c9c",
  ],
  "kernels_optimized": [
    "sha256": "16ba1df86a103ff02982f96d993406cc214ffdaf0ec1015bd43019fe00e2d4fc",
    "sha256" + debug_suffix: "dee61899f8fb28e34499b486f316990f8bf97e5fd1df3b2ffa940a95b9e94405",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "3ed0087c6cbdead21ada61a953767b665c2578487a5efc1f028df4e04e6caa56",
    "sha256" + debug_suffix: "48fb6f493044613a2641941e832c6ba2cd651f224afac94ed66c790da5aa2568",
  ],
  "kernels_torchao": [
    "sha256": "cb8838b9940e8d7965ab7855e4bffe747cbb57ad6af744262b423d0c644266d2",
    "sha256" + debug_suffix: "af74c465798fd0503c82ddbdfe4667ac3b32e9aac3153a7f5ebb735d14f23e91",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "bb2694a8d3c373b02aab64c856fcb585a8fe7467a07f75efa519ad36fc40cbee",
    "sha256" + debug_suffix: "f22fd139e584f824067339fbc76204d8ed04df99caf1dd9cfbef2e1d307856dd",
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
