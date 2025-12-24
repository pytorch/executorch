// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.1.0.20251224"
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
    "sha256": "e6b44a15ccb1ee50d4d092d2c0f2748e775e189b6b5591eecbbcfbe48f478333",
    "sha256" + debug_suffix: "3d6b716b7199931875958a69612d0a033f2e1da2317f8be5f5d9768da8d8d121",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "5a63005104406dd5bb8bfd1466297634898d7af52016eb6aaa01184a103cae10",
    "sha256" + debug_suffix: "5f6c51a25215adb2e0b56f65adcdfada72a96e8e76b2bbaf74903fa900b4dc65",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "6ccf2b9badc702a79d789155667e0019f0857875e3b3f4305c813b0c8c956789",
    "sha256" + debug_suffix: "eba869991851ef5ed4b009173dbabe015cce5756a7b35fa2546d4ad7c779e8e9",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "12247ade65439a7521201df19e29297b0990b30556e5af7a3dfd1633cdf7a903",
    "sha256" + debug_suffix: "12999b7982e234112f8632f02a47d6f17a7e00599ca298e35dce504fa38e0d54",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "2d5585712b590d1f3badc8b675db3ffdb4eeb7aad77bad3fe112134d0cd47e4a",
    "sha256" + debug_suffix: "f072fa3f809270a8cb8e3e35a129a055a9cca8601d1c255c802c38ad06910928",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "7d8add5fb4eabb4abaceecfc193ce02856fc2f2222526d1cfeb6d1d70d9328e5",
    "sha256" + debug_suffix: "46b6ba88682c5ede603fed5da87a2d677135feedfea1a38dfe77959270728412",
  ],
  "kernels_optimized": [
    "sha256": "c90aada83e96d630890c154f0bec522db9ee31223c83911cdee2a87f3893d345",
    "sha256" + debug_suffix: "acb2074a7a6e519db812c16b73aa362908bb042b4bbb6332504608612ce913a4",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "3f7554734e038e350ff8c5be1f5d25a2fda5023120b2acc5da4e8804ad4d32e1",
    "sha256" + debug_suffix: "073f51afd3fec7d24147906b18a3b85433925a083e3f488c399d8a7e70de42f4",
  ],
  "kernels_torchao": [
    "sha256": "cc4d6f903b084c9ae16879552b1c9b56f92f4a4b3bcb3b14c47dc096d44a913f",
    "sha256" + debug_suffix: "4ab1cf3e257d5c2d583b10ee2f5b327e8729d54451deb7b193a51d63ed48463d",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "4ead1b00ac8aca10b5b3611840b0f7f42c36cd1bd5266a77278d238d4a158e60",
    "sha256" + debug_suffix: "6aa54fe59e11bd9a2f2478a08d97ec7a76153f5cc4bf91f8640de7e730249fba",
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
