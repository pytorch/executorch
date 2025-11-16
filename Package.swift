// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251116"
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
    "sha256": "81940e16a67f140c8de1b81aae46a37e710fb8b004ed40bef24ce4522b95eafa",
    "sha256" + debug_suffix: "2136faf65a4f72a056bb93b345b831b15ff9032b32b21f1b8b3ff47a49baebac",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "09803f382aa98a8a5747a8273f302d84ac7ef655d293a25292aeb12387e47d4a",
    "sha256" + debug_suffix: "2d5521386a66f21a66b8597af55ac7c072cb128c1d0a85b18d3b0f1519786145",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "57f4f387a82bed13aed70f85cbc499cef42db199e2618981090b21f5991cdd5f",
    "sha256" + debug_suffix: "fe4f5a9264b2c55cfb25d86768b80ce5666c9ed76bfee91b9d82133626518e85",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "257f5c99d33a392062fc67a38ac5188b4230aaa417c3d08403223393bdd2ee37",
    "sha256" + debug_suffix: "4a6df0088466caddf45ba51fc3d297de4e0959705370ed8dacfe548f1b082cc0",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "8d602c680659b4642dcad2e65587558867401271673f9ff4f8894d234c3839d4",
    "sha256" + debug_suffix: "f1f27b2da1762aa12c142e70bacc7aff6a044f347568210f9ded1bf727cf96ea",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "7843f239eb1c560494f0b6ece5d116f8dbb42d37e5247f20e35804123da106f3",
    "sha256" + debug_suffix: "610e630a79812d15a4998167174bb3dff7e1d4f93a12d0c21fcf1c59eb6d060d",
  ],
  "kernels_optimized": [
    "sha256": "9b4ab76a4226a73fd0a43599a2d1fb449eb7c05df840e337679f36c14219c6da",
    "sha256" + debug_suffix: "dd4d2371bdde1da17645834903363b14c3c05ee0326c0f97995548a8a15e0513",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "8052ac67361eba4342186797f4b098ff077e2c6bb9bc75b434f810677cfa71f3",
    "sha256" + debug_suffix: "3a68a77183e386eb1db0b6b9d8629f1dab474ffd21e165473d8075af6259ecfc",
  ],
  "kernels_torchao": [
    "sha256": "bc0e57b8c0a5ea66682d42bf7f1f5ce8a259b1734089adc428a1c9fc2202791a",
    "sha256" + debug_suffix: "1ac8ffe8820eaefa529eaffee6e481da42d523d30d0a39294354542dc3cf3f3a",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "d91a145b39b6f8078cb3a58b5542276373e89abbed6b87da0b7031bb47bac8a0",
    "sha256" + debug_suffix: "0ba5da94bb787c78af055dbe3d204e5da5c5006165085243e6c4d96f4618f8ac",
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
