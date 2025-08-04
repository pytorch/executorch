// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250804"
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
    "sha256": "dc97eeaa69359d44a292e79f22ecbb618cf242e4665df2dc2acd40f9d58bb210",
    "sha256" + debug_suffix: "0f74d5489cfd00f1240295f40a6a13389f1f21244b40b3a6ce67a7f03fc41b5c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "619b599007571dd2619399d04db8eadca6091eded1c8700830e3fe6c4808c6cb",
    "sha256" + debug_suffix: "73da024b3732fb8726b64f4e3a5e06d75c5e5d0b62da237eeba462743ddabb8a",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "0bb442ef92cdd8e228498c77eec310430ffb25b054687f05762ca38e21d8efce",
    "sha256" + debug_suffix: "a398a0c7cd673d4fa5c56a393a6b83fb3fe0e5e62d22c55642e65ea58eb37e92",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "9a88b9379a431c8ba99bae8862b393e74dd5dc3de81f1623db69274cab92bfb1",
    "sha256" + debug_suffix: "a5fe31fec11191d83f622c6a4f8832ca86bdcfe96a63bd7bf907ffe2da0f2738",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "ba025a9b693e84adc86d8d5cd7dedb76718b7418015b42525295cdbdb3e63c98",
    "sha256" + debug_suffix: "80880422a3f5e840606fe54b61a702c2ede806f993011b1d357c940ec848b0a0",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "b514b25c816404a24d5ec6225714b2e2db42818343499d3332517757769ec712",
    "sha256" + debug_suffix: "ffa15c029fb87f048c42cf60a8892db20da0ace8c982687546641869edf56a71",
  ],
  "kernels_optimized": [
    "sha256": "7225492fc6a0c77d8f23973d5b5ba5dc236616bebd86a81e80902e2c89687151",
    "sha256" + debug_suffix: "80dc0c88a82809da894a7bbc687c11817b897d6ae121bdc38c106b8605d99a63",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "84922bb525ff431dcbbc2a596102f861a22747fd3e33a0335a240f1c01fe8ed6",
    "sha256" + debug_suffix: "3fc14d5850782d815829d4f490b39f7008a87c753f5b9cf38cce5bb4bd54922c",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "849e5b9fa4e01e7b89bb7aedbdcdde2bc22a5ea9da22bac67664b08029746332",
    "sha256" + debug_suffix: "6c90e6ee919c9aefc4078fdccad39eeb57a81f1b97e7b4ae73c8ed3aa30889d5",
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
