// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "0.8.0.20250906"
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
    "sha256": "9ce8e909b0e5970a967b95e84d5b9b9b70f999f15567ac526ca756cb2b160cd4",
    "sha256" + debug_suffix: "cc6d2ef3b4fc0c0a3724b262b14313885628e650e0a17748f8f245af096669ca",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "47309e5694427f06d0250aa7409d9536ea1d661e6f87da0ad584a63cca68e3f6",
    "sha256" + debug_suffix: "becf0b45ae4f8d39209808e256cdba8554b807868259c9a806787813b2995bc6",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "0197aea5a01e1ffb59cea42eed5d5149ad8c7c65e1367bac13ec5dbf075afa08",
    "sha256" + debug_suffix: "a59e12370f04489ca8abf2d03281a2e8ec9bb20c659a711f7b9ac8d2e11ae879",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "3cce2126dca039764c1567313bc414c1c90ea665c5aca93f617a874eb609090d",
    "sha256" + debug_suffix: "6f2951e3f0abd0d11de263e423914c4194d9e9424f6cfd8f099e47571308dd09",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "b65fe0e5c724ad6ce768cdf8228e60de8069483c6612b679df1bb44b9fe9244c",
    "sha256" + debug_suffix: "858c918c8420cae37b7443ff4d382e5e497bcd370f9b06613ae4436adf3e3b44",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "edb1c3e86aa7e0eab3201d00b4512195a6196bef60a563cf904a41d3510f4b88",
    "sha256" + debug_suffix: "ada087a20ee4ab6b65b8e4c1404d8b2da421bd0c55a9294c9b8820fb83cea25d",
  ],
  "kernels_optimized": [
    "sha256": "55b676643690536b1191b9712b8f796833fe687627779570b157ee0230174115",
    "sha256" + debug_suffix: "78ad73fe4f8be315fda159876fa0f37431e257cc29df5e3e851f2c3d92ae4fae",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "4dfef403ac61dd6fda93c5dc19b7294aaeb2fb73797fac2cab0af93276e7124e",
    "sha256" + debug_suffix: "702aca8d35144599912c252a9ac2bb35be2f6c9fdfe68edf9b2fa37a3d10cf54",
  ],
  "kernels_torchao": [
    "sha256": "0582e7bf09f4857b5a8ba2fbfb361ce14dcce3b34a0c7ea16c19d57c03f02064",
    "sha256" + debug_suffix: "0a6a6abd9123469070b9435bb8d3ef379c81ef1960191eae5be1152081b3ac15",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "d1e10c399eb4f3c5b0c0cc0fbefc2a1db65d54ebfa691208b91953b4c06c2a8d",
    "sha256" + debug_suffix: "d1b1c4385ba67b163af08c3ec4f530605c80a05b7929af4e570aa8bff300ea0f",
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
