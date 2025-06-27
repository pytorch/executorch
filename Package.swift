// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0.20250627"
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
    "sha256": "f88fcf6eb9ecf23b4e34d4f0f95f2bf8cfa45416960ef2c5948c180c6feda9d8",
    "sha256" + debug_suffix: "ece26ed7bb6e5e9a8068941f8b1777b04a4e58d0f2b4fff3c2c9e106f55735e9",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "a72891cd07be89a4b038b0cd540c155fda60cae18246da283508ca9ad91acd83",
    "sha256" + debug_suffix: "439deffeb2dc59391d3e395045a5102f1f9cad0f578eea2082a54b024759217b",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "ed8cb9b324dea31f88ae2d99a486ab4ebca97d09e5e0a6a35736e3024ad880aa",
    "sha256" + debug_suffix: "ac3f6e031025bde64aa023b8b247e15cc7952c7b065794ce63b930b2e37ab4d2",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "2ad3ff8d556e77d69c50d9d0c5c528bc8e3f715278a6946637f78826b782a687",
    "sha256" + debug_suffix: "46032f768212c121bf091c0b7f3aa3166fe65c160afdccd1cb6dcd8471744ca8",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "cf267f3435d884613167079833e306cee367d2cb6bc5afa39db9edf40bb48750",
    "sha256" + debug_suffix: "94ff38ac547c278d4e6c5dd01b2bc851c2066e26381f52c87fb7cf7b11a39c5e",
  ],
  "kernels_optimized": [
    "sha256": "ef9173cf0312fea2fa4cadf38fe5f85af54588775cb411c1b38808abe6e4d8a2",
    "sha256" + debug_suffix: "6e8a6004c1c584a89a3cc8b63743b00a178a37754b707b92151e17c6395bbaca",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "ec9909e2f15a5f7c2ea3888d100fcb5f23196430da859f5de15e7aea7eee1bf0",
    "sha256" + debug_suffix: "63f1925251d591e2e952ecfe242bc11ed90248c0a00dbfc8e2bbda6dd878f7f9",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "0bba83b49491cc24e7209a5bfdc67d01d689b7331d5b51e767a9085a823b41bd",
    "sha256" + debug_suffix: "cc263f018b24896cc2ad870bef510a3bf762fca6043120fb48fa0649e1bfa06d",
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
