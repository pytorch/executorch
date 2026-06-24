// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.4.0.20260624"
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
    "sha256": "d0b9cb762cb99706ea2d1e4130e7165a8946f57083af544cf77508e653f8c33c",
    "sha256" + debug_suffix: "d58d3c1bb15eae443920b2ba1fadcf12411b33dd91dbb92253121c650a3693ad",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "4a0d4e9b681f2657707d22f695ca2cc01c35417ba152b47629ca2c1fa24126bb",
    "sha256" + debug_suffix: "bd93096e5960ee7e381e31b7343bd58f35dc28e58a0d190431cab2911417c6a2",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "187ba3e22ba6fd87b80408fac3f782aed9fc4ffe2fef190417048083f5b5dd87",
    "sha256" + debug_suffix: "0442cbd6baa4cd50af7bbae271f0a9eaab731bbfe3b8e6bd181ae7def7443fef",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "1ce78c8898fc21b3f7a182eb0f1220eaf1d36f0fde8ee87e3fe7f71d69aeaf7a",
    "sha256" + debug_suffix: "e3cbe1cdff4c5115ab7bf61f96b34b1be1e3da6d459a1d400040294b11ef9c53",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "cb54258c08554b28026e15a073286a2fd3339425346366e2d50133b1d8720806",
    "sha256" + debug_suffix: "00b20d1c3797d831549248ecc14761573c33c0437c05c82101ccf0a650d2da75",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "8157cb406c5805759afd1215078db48c78639662ef57b608a755de138d7aed07",
    "sha256" + debug_suffix: "7bce8ac8c912c07814f78fc0eee1653dd833ff54c259528c03244d506776d07d",
  ],
  "kernels_optimized": [
    "sha256": "68fdda8b7dac38e00fa4745ecebd27152c4a945249b8b21dee147ee7dbc006ab",
    "sha256" + debug_suffix: "b3d3672015cc4e5b4850299e6a3773a2e6d36d8b1d29ed779ed7e96b59a92574",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "045aa780355d169289d9a25726b3b12fcaa805efe337d99d627d14b11d1b0b0a",
    "sha256" + debug_suffix: "c739ebee53e907c1ffa1d26ba1cfd25644609ae3dc0be366e6dd42883e806111",
  ],
  "kernels_torchao": [
    "sha256": "fd59ff0b9c2420cd0473512180bd49c542e8ae1c698e952f211d978c43ad272a",
    "sha256" + debug_suffix: "5dbbe88bbb14f22a8f27f45b71f2508503de4df31b9d057d9b19bd3a47f5b83b",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "f55babcc67362e5a9be272d461535e9460a107239d572d846fd415fde936665a",
    "sha256" + debug_suffix: "3ca8d2444c140964201a3d9089414441860aba5b734a8a118b0a59110079dcd4",
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
