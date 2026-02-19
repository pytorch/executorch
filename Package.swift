// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.2.0.20260219"
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
    "sha256": "9cda5640527e05f9186a670388ad7b23445abb635d7e61629d96ca54e9672c54",
    "sha256" + debug_suffix: "cd8525c3874f31d199a4d79d9615d210058e744dc82ef64483e25509dcfe6494",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "7d9f6d19a7e6c7dd5ff63e8c1cdd75b081ec6314c4a71f77cf0c0eb6db71717a",
    "sha256" + debug_suffix: "e6edfea5ebf34af3fd6771c5ddc3204da72f0a03c99c758b88e4a960bd31b338",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "eabd4785d983e199bcf0c0b751ed4887f7948e2031e242f4839c801d1e88564a",
    "sha256" + debug_suffix: "102723e58e747f18368965b2f00d66486e72df71c32405f345362e6324ae4fa4",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "001de3bb35ec6bbda1b591b65076db6b30503c7ed3785a93734ba77145f6600e",
    "sha256" + debug_suffix: "8b8d54c6363a3907b1f5448dba3ee2ae70ba996c7dbda3a6a4d2053c4d0b1692",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "14ca07b60bfa21b3d3ef30b9f346487cd9daed4d70317aa944524c50cff93256",
    "sha256" + debug_suffix: "72b5ba002876763970cee11ae8c451fe939aff06d5d54c97df9a12a884932238",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "4bd080dc87f3ff6611b03ff0159efd064c8e47bb2d60065ef8c6bdecf5d49a7a",
    "sha256" + debug_suffix: "81f0f1a70782e69ee8f4f1c40359ee30f7254c71588fde4601a25f5564be273a",
  ],
  "kernels_optimized": [
    "sha256": "eb4304eede3bb3583f7d5d215832171907d9ca63318f5fdb26545a173310021e",
    "sha256" + debug_suffix: "24f1fd5119ec01826e3f21ca6f06b22497b829330dd1cc38fc781581e6d85f57",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "4424f87d196f6e02ab5c20dfd6a75a378b8f0becea9f7920838e982419c112db",
    "sha256" + debug_suffix: "65619dd1d7e7b5d6362903a9e74838ff3ff8ebc8635e7830f82c9f6b3b2d017a",
  ],
  "kernels_torchao": [
    "sha256": "5cad795a48e113a6312a8964fba178208e23216dd7747484dd3debffc56702f6",
    "sha256" + debug_suffix: "3f825d077123d6d14995e65946b6ecdc5d4b7a52b3a51e4bb5a41e73e6e1f104",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "5dfcb41333917a3ed862bc42d0699692def680477124f13e4343ca008dc41bca",
    "sha256" + debug_suffix: "8d2043d548b50d905b2fa8b022afbadfbae489930f91c4313e358a122120825a",
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
