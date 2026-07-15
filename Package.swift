// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260715"
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
    "sha256": "2222ca919f334708e2acfe23bc4b23fe08b46c5cf76e98a8fe5f8ad0673e350a",
    "sha256" + debug_suffix: "f585393d23866c677de00e296e54ad764bd5fe26dabb6aec7ed1869de48e2938",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "e7fec149c2e62cef5753b93ca97f4bee644a74b56a704ff416c6d9e4bbd204b6",
    "sha256" + debug_suffix: "b71981bb6b56dfe2b2ac2ea2607af830b894b00948c4bb37e1a7af99dde52cee",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "438b8e68c855100075303dff018489e63e003e0c9b14f99d9d7365bf9a68d840",
    "sha256" + debug_suffix: "9e98cce2dff681065d5f32df09757601eb5c0ad66c7c1bd4772c9efbe9c9c2d5",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "24d814b975c3b2a6bb73915fdf4e8f20598a0a89ee04b561535f05df58d07ef8",
    "sha256" + debug_suffix: "ee9a4993d24ffe58883ef67a073727ea5508bac406f0281224919ca0cb0a5186",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "7e0a8ae351c2e771f9c4811d9cb10cf5b62694bc42c4ef2eeb4e34a86ddf8faf",
    "sha256" + debug_suffix: "c3a671c6d686a666d3087fd8c7fbd91be7950f6fc2cdf0876df5025ee5e4fe0e",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "945f15092fe81108afed1fbad167da058af9c923e91fcc7f96f397a255df9c99",
    "sha256" + debug_suffix: "e4e28fa21cdb9bbebfbd192b9e77772a73a2fd54918703d7090be09c494df16a",
  ],
  "kernels_optimized": [
    "sha256": "29051c3f7ef32b7f1f8259d0539abc9b7425a71a364513fda6617e211f88141e",
    "sha256" + debug_suffix: "03fe07952336b48d806256d35046872856c6920247bdeb7a5683fe931d152845",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "55128d40c8c82af121f42704a2fa1e35118e172857209d2e7b9340792fe78148",
    "sha256" + debug_suffix: "74a4fd924986aa3e6ce8c937f0e256d096aa44eb2bb08dd97322001407b45223",
  ],
  "kernels_torchao": [
    "sha256": "e6a56a455c2258c16dcae71744f7941725c83a039e4fae84e8e01b05bdb55ad7",
    "sha256" + debug_suffix: "47e1764eaa0c3bb5d5c978bd0ef1f4ce343eaa3e28691c08948daf90f9ac1491",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "e4d20564d4b59c8b3713e44ac8b0a8160472fe6916d0805056e9ecfbb31fef5b",
    "sha256" + debug_suffix: "45bde2efc85fcda1b244ba8042e709ffef84feb90062e805808ff5c483662030",
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
