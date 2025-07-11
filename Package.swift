// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.7.0"
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
    "sha256": "f0369980dde0ef5d26bfe4fea946d01092a0e7e6fd6e70c144d5265e958f8988",
    "sha256" + debug_suffix: "c493d984e475c884375ce9bf89c4c6abf62840fa394980bdcfd5ced487951d6c",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "76c1a40f5034491baaf4cedeed47e9a748f94a9375b61ab7c7e9ece0f4e00512",
    "sha256" + debug_suffix: "ad4ea28a53227adb09be2c600e4716aad47d94a9fedf37e9cbb364f526e359f1",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "8920fc84034fdffc02306b4fc36feb508c33b305f3b9ed37c52e343a2f8c1e13",
    "sha256" + debug_suffix: "3540e6182ccae9ca82f132558b6a6ce21070d5a7c1a70bd23b1398de320ce7af",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "e8c023711405e4107dce6795c0532139986116c5999fe164055c139387b4fedc",
    "sha256" + debug_suffix: "52015ea0ca5898fb52a44dbd7c6d0c454e31b08e404f112862f49960ef76c677",
    "libraries": [
      "c++",
    ],
  ],
  "kernels_custom": [
    "sha256": "aaf0458dcb04c4ed4a616bfb5675b1a8e6401de740512bef42504fe8efef803f",
    "sha256" + debug_suffix: "678a0ada408bdb0f50180b7094a67745ae8689df3c3827ed923dea1e1c46bfb0",
  ],
  "kernels_optimized": [
    "sha256": "f68d8d361edbbebcb3d550902b5599e6a09edf0f60e7fd45e901b55edc8067fc",
    "sha256" + debug_suffix: "182d89bd36b0d242af621c0a4e2c431c07ffbe6a914c41e79aa0df98c0cd0bd9",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "41d92a118da6a9cdc3afb43268cebf68a3550a95bf62c398f03f298e94af4b4d",
    "sha256" + debug_suffix: "4f8495de294b2cb81e011976babcfb302fb19eccfb9e4735f7dc93d318ab8417",
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "5b4c568f55a435b0f5ffe5c694cced673e1ce770505267194f671f3e68da14dd",
    "sha256" + debug_suffix: "35a5c18276ac30fb315e3d0d865a881d83daecbcc0f94e43cdfad7e4bc4de2a6",
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
